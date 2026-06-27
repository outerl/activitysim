# ActivitySim
# See full license in LICENSE.txt.
import os
from ast import literal_eval
import pandas as pd
import numpy as np
import pandas.testing as pdt

import activitysim.abm.models.school_escorting as school_escorting
from activitysim.abm.models.school_escorting import (
    SchoolEscortSettings,
    create_school_escorting_bundles_table,
    determine_escorting_participants,
)
from activitysim.abm.models.util import canonical_ids
from activitysim.abm.models.util.school_escort_tours_trips import (
    create_bundle_attributes,
    create_child_escorting_stops,
    create_chauf_trip_table,
    create_pure_school_escort_tours,
)


def test_create_bundle_attributes():
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    inbound_input = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_inbound__input.pkl")
    )
    inbound_expected = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_inbound__output.pkl")
    )

    outbound_input = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_outbound_cond__input.pkl")
    )
    outbound_expected = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_outbound_cond__output.pkl")
    )
    inbound_result = create_bundle_attributes(inbound_input)
    pdt.assert_frame_equal(inbound_result, inbound_expected, check_dtype=False)

    outbound_result = create_bundle_attributes(outbound_input)
    pdt.assert_frame_equal(outbound_result, outbound_expected, check_dtype=False)


def test_create_chauf_trip_table():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bundles = pd.read_pickle(
        os.path.join(data_dir, "create_chauf_trip_table__input.pkl")
    )
    chauf_trip_bundles = create_chauf_trip_table(bundles.copy())

    chauf_trip_bundles_expected = pd.read_pickle(
        os.path.join(data_dir, "create_chauf_trip_table__output.pkl")
    )

    pdt.assert_frame_equal(chauf_trip_bundles, chauf_trip_bundles_expected)


def test_create_child_escorting_stops():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bundles = pd.read_pickle(
        os.path.join(data_dir, "create_child_escorting_stops__input.pkl")
    )

    escortee_trips = []
    for escortee_num in range(0, int(bundles.num_escortees.max()) + 1):
        escortee_bundles = create_child_escorting_stops(bundles.copy(), escortee_num)
        escortee_trips.append(escortee_bundles)

    escortee_trips = pd.concat(escortee_trips)

    escortee_trips_expected = pd.read_pickle(
        os.path.join(data_dir, "create_child_escorting_stops__output.pkl")
    )

    pdt.assert_frame_equal(escortee_trips, escortee_trips_expected)


def _make_escorting_persons():
    """
    Two households exercising tie-breaks in chaperone/escortee numbering.

    Household 1: two parents with an identical chaperone_weight (same ptype,
    same sex, both > 25) and two escortees of the same age -- both are ties.
    Household 2: parents and escortees that differ, so the ranking is
    unambiguous and we can check the ordering is by weight / age.
    """
    persons = pd.DataFrame(
        {
            "person_id": [101, 102, 103, 104, 201, 202, 203, 204],
            "household_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "ptype": [1, 1, 8, 8, 1, 4, 8, 8],
            "sex": [1, 1, 2, 2, 1, 2, 1, 1],
            "age": [40, 40, 9, 9, 45, 42, 7, 12],
            "is_student": [False, False, True, True, False, False, True, True],
            "cdap_activity": ["M", "M", "M", "M", "M", "M", "M", "M"],
        }
    ).set_index("person_id")
    return persons


def _make_escorting_choosers():
    return pd.DataFrame(
        {"household_id": [1, 2], "home_zone_id": [5, 6]}
    ).set_index("household_id")


def _participant_assignments(persons):
    model_settings = SchoolEscortSettings(ALTS="dummy")
    choosers, participant_columns = determine_escorting_participants(
        _make_escorting_choosers(), persons, model_settings
    )
    return choosers[participant_columns]


def test_determine_escorting_participants_order_independent():
    """
    The chauffeur/escortee assignment must not depend on the set or order of
    rows in the persons table.

    Numbering is done with a single global sort over all chaperones (resp.
    escortees) in the chooser set, followed by a per-household cumcount. Because
    the default sort is non-stable, the relative order of two tied rows (e.g.
    two parents with an identical chaperone_weight) depends on the other rows in
    the array, not just on their own household. Multiprocessing partitions by
    household -- so a household's persons are always kept together and in
    person_id order -- but each process still sorts a *different set of other
    households*, which can flip a tied household's numbering between single- and
    multi-process runs. The deterministic person_id tie-break removes that.

    Reversing / interleaving the rows is a strong order-independence check; the
    single-household subsets emulate an actual MP slice (within-household order
    preserved, different set of other households present).
    """
    persons = _make_escorting_persons()
    baseline = _participant_assignments(persons)

    candidates = [
        persons.iloc[::-1],  # fully reversed
        persons.iloc[[3, 0, 2, 1, 7, 4, 6, 5]],  # interleaved
        persons[persons["household_id"] == 1],  # MP slice: household 1 only
        persons[persons["household_id"] == 2],  # MP slice: household 2 only
    ]
    for reordered in candidates:
        result = _participant_assignments(reordered)
        common = result.index.intersection(baseline.index)
        pdt.assert_frame_equal(
            result.loc[common].sort_index(), baseline.loc[common].sort_index()
        )


def test_determine_escorting_participants_tie_break_by_person_id():
    """Ties resolve to the lowest person_id getting the lowest number."""
    assignments = _participant_assignments(_make_escorting_persons())

    # Household 1: parents 101 & 102 tie -> 101 is chauf 1; escortees 103 & 104
    # tie on age -> 103 is child 1.
    assert assignments.loc[1, "chauf_id1"] == 101
    assert assignments.loc[1, "chauf_id2"] == 102
    assert assignments.loc[1, "child_id1"] == 103
    assert assignments.loc[1, "child_id2"] == 104


def test_determine_escorting_participants_ranking():
    """Chaperones rank by weight (desc) and escortees by age (asc)."""
    assignments = _participant_assignments(_make_escorting_persons())

    # Household 2: parent 202 (ptype 4) outweighs 201 (ptype 1) -> chauf 1.
    assert assignments.loc[2, "chauf_id1"] == 202
    assert assignments.loc[2, "chauf_id2"] == 201
    # Younger escortee (203, age 7) is child 1, older (204, age 12) is child 2.
    assert assignments.loc[2, "child_id1"] == 203
    assert assignments.loc[2, "child_id2"] == 204


def test_create_bundles_pickup_dropoff_order_stable():
    """
    Escortees with identical home-to-school times (e.g. siblings at the same
    school) must get a deterministic pickup/dropoff order -- by child number --
    rather than depending on the non-stable default argsort tie-break.
    """
    # create_school_escorting_bundles_table reads these module-level globals.
    school_escorting.NUM_ESCORTEES = 3
    school_escorting.NUM_CHAPERONES = 2

    choosers = pd.DataFrame(
        {
            "household_id": [100],
            "home_zone_id": [5],
            "nbundles": [1],
            "bundle1": [1],
            "bundle2": [1],
            "bundle3": [0],
            "child_id1": [1003],
            "child_id2": [1004],
            "child_id3": [0],
            "chauf1": [2],
            "chauf2": [2],
            "chauf3": [0],
            "chauf_id1": [1001],
            "chauf_id2": [1002],
            # children 1 & 2 attend the same school -> identical travel time (tie)
            "time_home_to_school1": [10.0],
            "time_home_to_school2": [10.0],
            "time_home_to_school3": [99.0],
            "alt": [2],
            "Description": ["test"],
        }
    ).set_index("household_id")

    tours = pd.DataFrame(
        {
            "tour_id": [50011, 50031, 50041],
            "person_id": [1001, 1003, 1004],
            "tour_type": ["work", "school", "school"],
            "tour_num": [1, 1, 1],
            "tour_category": ["mandatory", "mandatory", "mandatory"],
            "start": [9, 8, 8],
            "end": [17, 15, 15],
            "destination": [30, 20, 21],
            "origin": [5, 5, 5],
        }
    ).set_index("tour_id")

    bundles = create_school_escorting_bundles_table(choosers, tours, "outbound_cond")

    # tied times -> stable order is child-number order [1, 2, 3]
    assert list(bundles["child_order"].iloc[0]) == [1, 2, 3]
    # escortees string is built in that order: child 1003 before child 1004
    assert bundles["escortees"].iloc[0] == "1003_1004"


class _FakeState:
    """Minimal stand-in for workflow.State used by create_pure_school_escort_tours."""

    def __init__(self, tours):
        self._tours = tours

    def get_dataframe(self, name):
        assert name == "tours"
        return self._tours


def _fake_set_tour_index(state, tours, is_school_escorting=False, **kwargs):
    # mirror the real id property: stable in person_id and tour_type_num, so an
    # order-dependent numbering would still surface as a different tour_id here.
    tours["tour_id"] = tours["person_id"] * 100 + tours["tour_type_num"]
    tours.set_index("tour_id", inplace=True)
    return tours


def _make_pure_escort_bundles():
    """
    One chauffeur (1001) with two outbound pure-escort bundles that share the
    same start time -- a tie on (household_id, person_id, start).
    """
    return pd.DataFrame(
        {
            "bundle_id": [1001, 1002],
            "household_id": [100, 100],
            "chauf_id": [1001, 1001],
            "escort_type": ["pure_escort", "pure_escort"],
            "school_escort_direction": ["outbound", "outbound"],
            "home_zone_id": [5, 5],
            "school_destinations": ["20", "21"],
            "school_starts": ["8", "8"],  # identical start -> tie
            "school_ends": ["15", "15"],
        }
    )


def _run_pure_escort(bundles, monkeypatch):
    monkeypatch.setattr(canonical_ids, "set_tour_index", _fake_set_tour_index)
    tours_for_dtype = pd.DataFrame(
        {
            "tour_category": pd.Categorical(["non_mandatory"]),
            "tour_type": pd.Categorical(["escort"]),
        }
    )
    state = _FakeState(tours_for_dtype)
    result = create_pure_school_escort_tours(state, bundles.copy())
    return result.reset_index().set_index("bundle_id")


def test_create_pure_school_escort_tours_tie_order_independent(monkeypatch):
    """
    Two pure-escort tours for the same chauffeur with an equal start time must
    get a reproducible ordering regardless of input row order. Without the
    bundle_id tiebreaker the non-stable sort orders the tied pair by incoming
    row order, flipping tour_num / next_pure_escort_start / tour_id between
    single- and multi-process runs.
    """
    bundles = _make_pure_escort_bundles()
    baseline = _run_pure_escort(bundles, monkeypatch)
    reversed_in = _run_pure_escort(bundles.iloc[::-1], monkeypatch)

    cols = ["tour_num", "tour_type_num", "next_pure_escort_start"]
    pdt.assert_frame_equal(
        reversed_in[cols].sort_index(), baseline[cols].sort_index()
    )
    pdt.assert_series_equal(
        reversed_in.index.to_series().sort_values(),
        baseline.index.to_series().sort_values(),
    )


def test_create_pure_school_escort_tours_tie_break_by_bundle_id(monkeypatch):
    """The lower bundle_id sorts first, so it becomes tour 1."""
    assignments = _run_pure_escort(_make_pure_escort_bundles(), monkeypatch)

    assert assignments.loc[1001, "tour_num"] == 1
    assert assignments.loc[1002, "tour_num"] == 2
    # first tour points to the next (equal) start; last tour has no next -> 0
    assert assignments.loc[1001, "next_pure_escort_start"] == 8
    assert assignments.loc[1002, "next_pure_escort_start"] == 0


if __name__ == "__main__":
    test_create_bundle_attributes()
    test_create_chauf_trip_table()
    test_create_child_escorting_stops()
    test_determine_escorting_participants_order_independent()
    test_determine_escorting_participants_tie_break_by_person_id()
    test_determine_escorting_participants_ranking()
    test_create_bundles_pickup_dropoff_order_stable()
