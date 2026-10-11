import json

import pandas as pd
import pytest

from src.schemas.common import CaseFilters
from src.services import data_prep


def test_row_count_matches_report(store):
    rep = json.loads(open("data/processed/data_quality_report.json", encoding="utf-8").read())
    assert len(store) == rep["final_row_count"] == 55000
    assert rep["real_row_count"] == rep["valid_row_count"] == 19689 == int((~store.df["is_synthetic"]).sum())
    assert rep["synthetic_row_count"] == 35311 == int(store.df["is_synthetic"].sum())
    assert rep["target_met"] is True and rep["target_met_with_real_data_only"] is False   # honest about how 55k is reached


def test_case_ids_unique_and_traceable(store):
    assert store.df["case_id"].is_unique
    syn = store.df["is_synthetic"]
    assert store.df.loc[~syn, "case_id"].str.startswith("TM-").all() and store.df.loc[syn, "case_id"].str.startswith("TS-").all()
    assert store.df["source_row"].is_unique and (store.df.loc[syn, "source_row"] < 0).all() and (store.df.loc[~syn, "source_row"] >= 0).all()


def test_no_missing_treated_as_zero(store):
    for c in ["n", "p", "k", "ph", "rainfall", "temperature", "humidity"]:
        assert store.df[c].notna().all()          # real values, no silent zero fill
    assert store.df["yield"].min() >= 0


def test_prepare_small_pipeline(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"crop": ["Rice", "Rice", "Wheat", "Rice"], "year": [2000, 2000, 2001, 2001],
                  "season": ["Kharif     ", "Kharif     ", "Rabi       ", "Kharif     "], "state": ["Assam"] * 4,
                  "area": [10, 10, 20, 0], "production": [20, 20, 40, 5], "fertilizer": [100, 100, 200, 5],
                  "pesticide": [1, 1, 2, 1], "yield": [2.0, 2.0, 2.0, 1.0]}).to_csv(raw / "crop_yield.csv", index=False)
    rep = data_prep.prepare(raw_dir=raw, out_dir=tmp_path / "out", crop_file=raw / "crop_yield.csv", target_rows=55000, augment=False)
    assert rep["source_row_count"] == 4
    assert rep["dropped_invalid"] == 1                    # area 0
    assert rep["dropped_exact_duplicates"] == 1
    assert rep["final_row_count"] == 2 and rep["target_met"] is False
    out = pd.read_csv(tmp_path / "out" / "cases.csv")
    assert out["n"].isna().all()                          # soil file missing -> missing, not zero


def test_prepare_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_prep.prepare(raw_dir=tmp_path, out_dir=tmp_path, crop_file=tmp_path / "nope.csv")


def test_prepare_samples_down_to_target(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [{"crop": f"C{i % 3}", "year": 1997 + i % 20, "season": "Kharif", "state": f"S{i % 5}", "area": 10 + i, "production": 20 + i,
             "fertilizer": 10, "pesticide": 1, "yield": 2.0} for i in range(300)]
    pd.DataFrame(rows).drop_duplicates(["crop", "year", "season", "state"]).to_csv(raw / "crop_yield.csv", index=False)
    rep = data_prep.prepare(raw_dir=raw, out_dir=tmp_path / "o1", crop_file=raw / "crop_yield.csv", target_rows=50, seed=1)
    rep2 = data_prep.prepare(raw_dir=raw, out_dir=tmp_path / "o2", crop_file=raw / "crop_yield.csv", target_rows=50, seed=1)
    assert rep["final_row_count"] == rep2["final_row_count"] <= 50
    assert pd.read_csv(tmp_path / "o1" / "cases.csv")["case_id"].tolist() == pd.read_csv(tmp_path / "o2" / "cases.csv")["case_id"].tolist()


def test_metadata_filters(store):
    f = CaseFilters(crop=["rice"], state=["assam"], ph_min=5, ph_max=6, year_min=2005)
    m = store.mask(f)
    sub = store.df[m]
    assert len(sub) > 0 and (sub["crop"] == "Rice").all() and (sub["state"] == "Assam").all() and (sub["year"] >= 2005).all()
    assert sub["ph"].between(5, 6).all()


def test_numeric_filter_is_inclusive(store):
    v = float(store.df["rainfall"].iloc[0])
    m = store.mask(CaseFilters(rainfall_min=v, rainfall_max=v))
    assert m.sum() >= 1 and store.df[m]["rainfall"].eq(v).all()


def test_invalid_filter_range_rejected():
    with pytest.raises(ValueError):
        CaseFilters(ph_min=8, ph_max=5)
    with pytest.raises(ValueError):
        CaseFilters(ph_min=1)                              # below plausible range


def test_filter_no_match(store):
    assert store.mask(CaseFilters(crop=["Coconut"], state=["Delhi"])).sum() == 0


def test_synthetic_augmentation_is_labelled_and_bounded(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [{"crop": f"C{i % 3}", "year": 1997 + i % 10, "season": "Kharif", "state": f"S{i % 2}", "area": 10 + i, "production": 20 + i,
             "fertilizer": 10, "pesticide": 1, "yield": 2.0} for i in range(60)]
    pd.DataFrame(rows).drop_duplicates(["crop", "year", "season", "state"]).to_csv(raw / "crop_yield.csv", index=False)
    rep = data_prep.prepare(raw_dir=raw, out_dir=tmp_path / "o", crop_file=raw / "crop_yield.csv", target_rows=200, seed=3)
    out = pd.read_csv(tmp_path / "o" / "cases.csv")
    assert rep["final_row_count"] == 200 and rep["synthetic_row_count"] == 200 - rep["real_row_count"] > 0
    syn = out[out["is_synthetic"]]
    assert syn["case_id"].str.startswith("TS-").all() and out["case_id"].is_unique and syn["flags"].str.contains("synthetic").all()
    real_pairs = set(zip(out[~out["is_synthetic"]]["crop"], out[~out["is_synthetic"]]["state"]))
    assert set(zip(syn["crop"], syn["state"])) <= real_pairs            # no invented crop/state pairs
    assert rep["target_met_with_real_data_only"] is False and "SYNTHETIC" in " ".join(rep["decisions"])
    rep2 = data_prep.prepare(raw_dir=raw, out_dir=tmp_path / "o2", crop_file=raw / "crop_yield.csv", target_rows=200, seed=3)
    assert pd.read_csv(tmp_path / "o2" / "cases.csv")["case_id"].tolist() == out["case_id"].tolist()    # reproducible


def test_synthetic_excluded_from_statistics(store):
    assert store.real_df["is_synthetic"].sum() == 0 and len(store.real_df) == 19689
    from src.schemas.common import CaseFilters
    assert store.mask(CaseFilters(origin=["real"])).sum() == 19689 and store.mask(CaseFilters(origin=["synthetic"])).sum() == 35311
