from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import openmatrix as omx
import pandas as pd

from activitysim.abm.models.trip_matrices import (
    MatrixSettings,
    MatrixTableSettings,
    WriteTripMatricesSettings,
    write_matrices,
)
from activitysim.core import workflow


def get_settings(omx_file_name):
    # Settings: same table name twice, second time with custom destination column
    settings = WriteTripMatricesSettings(
        MATRICES=[
            MatrixSettings(
                file_name=Path(omx_file_name),
                tables=[
                    MatrixTableSettings(
                        name="DRIVEALONE_VOT1",
                        data_field="DRIVEALONE_VOT1",
                    ),
                    MatrixTableSettings(
                        name="DRIVEALONE_VOT1",
                        data_field="PNR_DRIVEALONE_OUT",
                        origin="origin",
                        destination="pnr_zone_id",
                    ),
                    MatrixTableSettings(
                        name="WALK_TRN_OUT",
                        data_field="WALK_TRN_OUT",
                        origin="origin",
                        destination="destination",
                    ),
                    MatrixTableSettings(
                        name="WALK_TRN_OUT",
                        data_field="PNR_TRN_OUT",
                        origin="pnr_zone_id",
                        destination="destination",
                    ),
                ],
            )
        ]
    )
    return settings


def test_write_matrices_one_zone():
    state = workflow.State.make_default(__file__)

    # Zones domain (TAZ)
    zone_index = pd.Index([1, 2, 3, 4], name="TAZ")

    # Trips with both standard drive and PNR drive legs
    trips_df = pd.DataFrame(
        {
            "origin": [1, 1, 2],
            "destination": [2, 2, 4],
            "pnr_zone_id": [3, -1, 4],
            # we have more flags below than trips, but that's ok for testing
            "DRIVEALONE_VOT1": [0, 1, 1],
            "PNR_DRIVEALONE_OUT": [1, 0, 1],
            "WALK_TRN_OUT": [1, 0, 1],
            "PNR_TRN_OUT": [1, 0, 1],
            "sample_rate": [0.5, 0.5, 0.5],
        }
    )

    # Run writer
    write_matrices(
        state=state,
        trips_df=trips_df,
        zone_index=zone_index,
        model_settings=get_settings("trips_one_zone.omx"),
    )

    # Validate output
    omx_path = os.path.join(os.path.dirname(__file__), "output", "trips_one_zone.omx")
    assert os.path.exists(omx_path), "OMX file not created"

    with omx.open_file(omx_path, mode="r") as f:
        # Mapping should reflect zone_index order [1,2,3,4]
        mapping = f.mapping("TAZ")
        assert list(mapping) == [1, 2, 3, 4]

        # --- checking driving output
        assert "DRIVEALONE_VOT1" in f.list_matrices(), "Expected matrix not found"
        data = f["DRIVEALONE_VOT1"][:]

        # Expected totals:
        # - zones (1->2): 1 drive = 1
        # - zones (1->3): 1 PNR drive leg = 1
        # - zones (2->4): 1 PNR drive leg + 1 drive = 2
        # and double them based on sample rate
        assert data.shape == (4, 4)
        # indices are zero-based positions for labels [1,2,3,4]
        assert data[0, 1] == 2  # 1->2
        assert data[0, 2] == 2  # 1->3
        assert data[1, 3] == 4  # 2->4
        # everything else remains zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 1] = False
        zero_mask[0, 2] = False
        zero_mask[1, 3] = False
        assert np.all(data[zero_mask] == 0.0)

        # ---- checking transit output
        assert "WALK_TRN_OUT" in f.list_matrices(), "Expected matrix not found"
        data = f["WALK_TRN_OUT"][:]

        # Expected totals:
        # - zones (1->2): 1 wlk trn
        # - zones (2->4): 1 wlk trn
        # - zones (3->2): 1 PNR trn leg
        # - zones (4->4): 1 PNR trn leg
        # and double them based on sample rate
        assert data.shape == (4, 4)
        assert data[0, 1] == 2  # 1->2
        assert data[1, 3] == 2  # 2->4
        assert data[2, 1] == 2  # 3->2
        assert data[3, 3] == 2  # 4->4
        # everything else remains zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 1] = False
        zero_mask[1, 3] = False
        zero_mask[2, 1] = False
        zero_mask[3, 3] = False
        assert np.all(data[zero_mask] == 0.0)


# ...existing code...
def test_write_matrices_two_zone():
    state = workflow.State.make_default(__file__)

    # land_use table for MAZ -> TAZ mapping
    # MAZ: 101,102 -> TAZ 1; 103,104 -> TAZ 2
    land_use = pd.DataFrame(
        {"TAZ": [1, 1, 2, 2]},
        index=pd.Index([101, 102, 103, 104], name="MAZ"),
    )
    state.add_table("land_use", land_use)

    # TAZ domain for output
    zone_index = pd.Index([1, 2], name="TAZ")

    # Trips in MAZ space, with both standard drive and PNR drive legs
    trips_df = pd.DataFrame(
        {
            "origin": [101, 102, 103],
            "destination": [102, 104, 104],
            "pnr_zone_id": [103, 0, 104],
            # we have more flags below than trips, but that's ok for testing
            "DRIVEALONE_VOT1": [0, 1, 1],
            "PNR_DRIVEALONE_OUT": [1, 0, 1],
            "WALK_TRN_OUT": [1, 0, 1],
            "PNR_TRN_OUT": [1, 0, 1],
            "sample_rate": [0.5, 0.5, 0.5],
        }
    )

    # Run writer
    write_matrices(
        state=state,
        trips_df=trips_df,
        zone_index=zone_index,
        model_settings=get_settings("trips_two_zone.omx"),
    )

    # Validate output
    omx_path = os.path.join(os.path.dirname(__file__), "output", "trips_two_zone.omx")
    assert os.path.exists(omx_path), "OMX file not created"

    with omx.open_file(omx_path, mode="r") as f:
        mapping = f.mapping("TAZ")
        assert list(mapping) == [1, 2]

        # ---- checking drive output
        assert "DRIVEALONE_VOT1" in f.list_matrices(), "Expected matrix not found"
        data = f["DRIVEALONE_VOT1"][:]

        # Expected before expansion weighting:
        # Standard drive (MAZ->TAZ):
        # 102->104 => 1->2 : 1
        # 103->104 => 2->2 : 1
        # PNR drive legs (origin->pnr_zone_id):
        # 101->103 => 1->2 : +1
        # 103->104 => 2->2 : +1
        # Totals: (1,2)=2, (2,2)=2
        # With sample_rate=0.5, values are doubled:
        assert data.shape == (2, 2)
        assert data[0, 1] == 4  # 1->2
        assert data[1, 1] == 4  # 2->2

        # All other cells zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 1] = False
        zero_mask[1, 1] = False
        assert np.all(data[zero_mask] == 0.0)

        # ---- checking transit output
        assert "WALK_TRN_OUT" in f.list_matrices(), "Expected matrix not found"
        data = f["WALK_TRN_OUT"][:]

        # Expected before expansion weighting:
        # regular transit
        # 101->102 => 1->1 : 1
        # 103->104 => 2->2 : 1
        # pnr transit legs
        # 103->102 => 2->1 : +1
        # 104->104 => 2->2 : +1
        # Totals: (1,1)=1, (2,1)=1, (2,2)=2
        # With sample_rate=0.5, values are doubled:
        assert data.shape == (2, 2)
        assert data[0, 0] == 2  # 1->1
        assert data[1, 0] == 2  # 2->1
        assert data[1, 1] == 4  # 2->2

        # All other cells zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 0] = False
        zero_mask[1, 0] = False
        zero_mask[1, 1] = False
        assert np.all(data[zero_mask] == 0.0)


def _reference_matrices(trips_df, zone_index, settings, land_use=None):
    """Per-table group-by aggregation, as `write_matrices` did after PR 1001."""
    n = len(zone_index)
    matrices = {}
    for table in settings.MATRICES[0].tables:
        if table.data_field not in trips_df.columns:
            continue
        o, d = trips_df[table.origin], trips_df[table.destination]
        if land_use is not None:
            o = o if o.isin(zone_index).all() else o.map(land_use["TAZ"])
            d = d if d.isin(zone_index).all() else d.map(land_use["TAZ"])
        work = pd.DataFrame(
            {
                "_o": o,
                "_d": d,
                "v": trips_df[table.data_field],
                "w": trips_df["sample_rate"],
            }
        ).dropna(subset=["_o", "_d"])
        grouped = work.groupby(["_o", "_d"], sort=False)
        vals = (grouped["v"].sum() / grouped["w"].mean().replace(0, np.nan)).fillna(0.0)
        oi = zone_index.get_indexer(vals.index.get_level_values(0))
        di = zone_index.get_indexer(vals.index.get_level_values(1))
        mask = (oi != -1) & (di != -1)
        data = matrices.setdefault(table.name, np.zeros((n, n)))
        data[oi[mask], di[mask]] += vals.to_numpy()[mask]
    return matrices


def _random_trips(rng, origins, n=5000):
    labels = np.asarray(origins, dtype=float)
    origin = rng.choice(labels, n)
    destination = rng.choice(labels, n)
    origin[rng.random(n) < 0.02] = np.nan  # missing
    destination[rng.random(n) < 0.02] = -1  # outside the zone domain
    pnr = np.where(rng.random(n) < 0.3, rng.choice(labels, n), -1)
    sample_rate = rng.choice([0.1, 0.2, 0.3, 0.7], n)
    # all trips between the first two zones have zero weight
    sample_rate[(origin == labels[0]) & (destination == labels[1])] = 0.0
    flt = rng.random(n) * 3.3
    flt[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame(
        {
            "origin": origin,
            "destination": destination,
            "pnr_zone_id": pnr,
            "DRIVEALONE_VOT1": rng.random(n) < 0.4,  # bool
            "PNR_DRIVEALONE_OUT": rng.integers(0, 3, n),  # int
            "WALK_TRN_OUT": flt,  # float with missing values
            "PNR_TRN_OUT": rng.random(n) * 0.1,  # float
            "sample_rate": sample_rate,
        }
    )


def _assert_matches_reference(state, trips_df, zone_index, omx_name, land_use=None):
    settings = get_settings(omx_name)
    write_matrices(
        state=state, trips_df=trips_df, zone_index=zone_index, model_settings=settings
    )
    expected = _reference_matrices(trips_df, zone_index, settings, land_use)
    omx_path = os.path.join(os.path.dirname(__file__), "output", omx_name)
    with omx.open_file(omx_path, mode="r") as f:
        assert sorted(f.list_matrices()) == sorted(expected)
        for name, data in expected.items():
            # identical, not just close: same sums and the same mean weights
            np.testing.assert_array_equal(f[name][:], data, err_msg=name)


def test_write_matrices_matches_per_table_groupby_one_zone():
    state = workflow.State.make_default(__file__)
    zone_index = pd.Index([10, 20, 30, 40, 50, 60], name="TAZ")
    trips_df = _random_trips(np.random.default_rng(42), zone_index)
    _assert_matches_reference(state, trips_df, zone_index, "trips_random_one.omx")


def test_write_matrices_matches_per_table_groupby_two_zone():
    state = workflow.State.make_default(__file__)
    # MAZ 101-106 -> TAZ 1-3; MAZ 107 has no TAZ
    land_use = pd.DataFrame(
        {"TAZ": [1, 1, 2, 2, 3, 3, np.nan]},
        index=pd.Index([101, 102, 103, 104, 105, 106, 107], name="MAZ"),
    )
    state.add_table("land_use", land_use)
    zone_index = pd.Index([1, 2, 3], name="TAZ")
    trips_df = _random_trips(np.random.default_rng(7), land_use.index)
    _assert_matches_reference(
        state, trips_df, zone_index, "trips_random_two.omx", land_use
    )
