# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import json
import multiprocessing as mp
import secrets

import numpy as np
import openmatrix
import pandas as pd
import pytest
import sharrow as sh
import xarray as xr
import yaml

from activitysim.core import los, skim_dataset, workflow


@pytest.fixture(params=["omx", "parquet"])
def skim_cache_env(tmp_path, request, monkeypatch):
    zones = np.array([10, 20, 30])
    distance = np.arange(9, dtype=np.float32).reshape(3, 3) * 0.25
    distance[1, 1] = -999
    fare = np.array([[0, 1, 2], [1, 2, np.nan], [2, 1, 0]], dtype=np.float32)
    matrices = {
        "DIST": distance,
        "FARE": fare,
        "TIME__AM": np.arange(9, dtype=np.float32).reshape(3, 3),
        "TIME__PM": np.arange(9, dtype=np.float32).reshape(3, 3) * 2,
    }
    source = tmp_path / f"skims.{request.param}"
    if request.param == "omx":
        with openmatrix.open_file(str(source), mode="w") as out:
            for name, values in matrices.items():
                out[name] = values
            out.create_mapping("zone_id", zones)
    else:
        frame = pd.DataFrame({"orig": np.repeat(zones, 3), "dest": np.tile(zones, 3)})
        for name, values in matrices.items():
            frame[name] = values.ravel()
        frame.to_parquet(source, index=False)

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "spec.csv").write_text("Expression\nDIST\nFARE\nTIME\n")
    settings = {
        "zone_system": 1,
        "taz_skims": {
            "omx": source.name,
            "zarr": "skims.zarr",
            "zarr-digital-encoding": [
                {
                    "name": "DIST",
                    "scale": 0.25,
                    "offset": 0,
                    "missing_value": -999,
                    "bitwidth": 16,
                },
                {"name": "FARE", "by_dict": True, "bitwidth": 8},
            ],
        },
        "skim_time_periods": {
            "time_window": 1440,
            "period_minutes": 60,
            "periods": [0, 12, 24],
            "labels": ["AM", "PM"],
        },
    }
    (configs / "network_los.yaml").write_text(yaml.safe_dump(settings))
    state = (
        workflow.State()
        .initialize_filesystem(
            working_dir=tmp_path,
            configs_dir=configs,
            data_dir=tmp_path,
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache",
        )
        .default_settings()
    )
    state.settings.sharrow = True
    state.settings.store_skims_in_shm = False
    state.add_table(
        "land_use",
        pd.DataFrame(
            {"_original_zone_id": zones}, index=pd.Index(range(3), name="zone_id")
        ),
    )
    preload = los.Network_LOS(state)
    token = "zarr_migration_" + secrets.token_hex(8)
    monkeypatch.setattr(preload, "skim_backing_store", lambda skim_tag: token)
    state.set("network_los_preload", preload)
    expected = xr.Dataset(
        {
            "DIST": (("otaz", "dtaz"), distance),
            "FARE": (("otaz", "dtaz"), fare),
            "TIME": (
                ("otaz", "dtaz", "time_period"),
                np.stack([matrices["TIME__AM"], matrices["TIME__PM"]], axis=-1),
            ),
        },
        coords={"otaz": range(3), "dtaz": range(3), "time_period": ["AM", "PM"]},
    )
    yield state, tmp_path / "cache" / "skims.zarr", expected, token
    sh.shared_memory.release_shared_memory(token)


def _assert_skims(dataset, expected):
    decoded = dataset.digital_encoding.strip(["DIST", "FARE"])
    xr.testing.assert_allclose(decoded, expected)
    assert dataset.DIST.dtype == np.int16
    assert dataset.FARE.dtype == np.uint8
    assert dataset.attrs["ZARR_WRITE_TIME"] > 0
    wrapper = skim_dataset.SkimDataset(dataset).wrap("orig", "dest")
    wrapper.set_df(pd.DataFrame({"orig": [0, 1, 1], "dest": [1, 1, 2]}))
    np.testing.assert_array_equal(wrapper["DIST"], [0.25, -999, 1.25])
    np.testing.assert_array_equal(wrapper["FARE"], [1, 2, np.nan])


@pytest.mark.parametrize("shared", [False, True])
def test_encoded_cache_creation_and_reuse(skim_cache_env, monkeypatch, shared):
    state, cache, expected, _ = skim_cache_env
    state.settings.store_skims_in_shm = shared
    result = skim_dataset.load_skim_dataset_to_shared_memory(state)
    _assert_skims(result, expected)
    assert result.shm.is_shared_memory == shared
    assert json.loads((cache / ".zgroup").read_text())["zarr_format"] == 2
    assert not (cache / "zarr.json").exists()
    write_time = result.attrs["ZARR_WRITE_TIME"]
    result.shm.release_shared_memory()

    def fail_source_load(*args, **kwargs):
        pytest.fail("A valid cache should bypass the source loader")

    monkeypatch.setattr(
        skim_dataset, "_load_skim_dataset_from_sources", fail_source_load
    )
    recovered = skim_dataset.load_skim_dataset_to_shared_memory(state)
    _assert_skims(recovered, expected)
    assert recovered.attrs["ZARR_WRITE_TIME"] == write_time


@pytest.mark.parametrize("shared", [False, True])
def test_legacy_format2_cache(skim_cache_env, monkeypatch, shared):
    state, cache, expected, _ = skim_cache_env
    state.settings.store_skims_in_shm = shared
    legacy = expected.assign_coords(otaz=[10, 20, 30], dtaz=[10, 20, 30]).copy(
        deep=True
    )
    legacy["DIST"] = xr.where(legacy.DIST == -999, -1, legacy.DIST / 0.25).astype(
        np.int16
    )
    distance_encoding = " {'scale': 0.25, 'offset': 0, 'missing_value': -999} "
    legacy.DIST.attrs["digital_encoding"] = distance_encoding
    legacy["FARE"] = legacy.FARE.fillna(3).astype(np.uint8)
    legacy.FARE.attrs["digital_encoding"] = " {'dictionary': [0.0, 1.0, 2.0, nan]} "
    source = state.filesystem.expand_input_file_list(
        state.get_injectable("network_los_preload").omx_file_names("taz")
    )[0]
    legacy.attrs["ZARR_WRITE_TIME"] = source.stat().st_mtime + 1
    legacy.attrs["settings"] = " {'enabled': True, 'optional': None} "
    # Reproduce the format and attribute representation of existing caches.
    legacy.to_zarr(cache, zarr_format=2)

    def fail_source_load(*args, **kwargs):
        pytest.fail("An existing format 2 cache should bypass the source loader")

    monkeypatch.setattr(
        skim_dataset, "_load_skim_dataset_from_sources", fail_source_load
    )
    recovered = skim_dataset.load_skim_dataset_to_shared_memory(state)
    _assert_skims(recovered, expected)
    assert recovered.attrs["settings"] == {"enabled": True, "optional": None}


def _read_shared_skims(token):
    shared = sh.Dataset.shm.from_shared_memory(token)
    decoded = shared.digital_encoding.strip(["DIST", "FARE"])
    return decoded.DIST.values, decoded.FARE.values, decoded.TIME.values


@pytest.mark.parametrize("start_method", mp.get_all_start_methods())
def test_cached_skims_shared_between_processes(skim_cache_env, start_method):
    state, _, expected, token = skim_cache_env
    # Create the cache, then start a fresh shared-memory load from it.
    skim_dataset.load_skim_dataset_to_shared_memory(state)
    state.settings.store_skims_in_shm = True
    shared = skim_dataset.load_skim_dataset_to_shared_memory(state)
    _assert_skims(shared, expected)
    with mp.get_context(start_method).Pool(2) as pool:
        results = pool.map_async(_read_shared_skims, [token, token]).get(timeout=60)
    for distance, fare, time in results:
        np.testing.assert_array_equal(distance, expected.DIST.values)
        np.testing.assert_array_equal(fare, expected.FARE.values)
        np.testing.assert_array_equal(time, expected.TIME.values)
