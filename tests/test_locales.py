"""Per-locale audits: find a workbook's locales (a tab per locale, or a locale column) and resolve one to rows."""
import pandas as pd
import pytest

from src.qa.locales import LocaleError, list_locales, resolve


def _tab(tasks, assigned, empty=2):
    rows = [{"task_id": f"t{i}", "Task Assignee": f"a{i}@x.com" if i < assigned else None,
             "AR Rationale": f"The verdict holds because point {i} is covered." if i < assigned else None}
            for i in range(tasks)]
    rows += [{"task_id": None, "Task Assignee": None, "AR Rationale": None}] * empty
    return pd.DataFrame(rows)


def _write(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return path


def _main(locales):
    return pd.DataFrame([{"task_id": f"m{i}", "locale": loc, "Rating_human_rationale": f"Rationale {i} for this row."}
                         for i, loc in enumerate(locales)])


def test_each_locale_tab_is_listed_with_its_task_and_assigned_counts(tmp_path):
    path = _write(tmp_path / "wb.xlsx", {"Main sheet": _main(["en_US"] * 3 + ["hi_IN"] * 3),
                                         "en_US": _tab(3, 2), "hi_IN": _tab(3, 0), "Readme": pd.DataFrame({"x": [1]})})
    got = {e["locale"]: e for e in list_locales(path)}
    assert set(got) == {"en_US", "hi_IN"}
    assert got["en_US"] == {"locale": "en_US", "source": "tab", "sheet": "en_US", "rows": 3, "assigned": 2}
    assert got["hi_IN"]["assigned"] == 0


def test_a_locale_column_is_used_when_there_are_no_locale_tabs(tmp_path):
    path = _write(tmp_path / "wb.xlsx", {"Sheet1": _main(["en_US", "en_US", "fr_FR"])})
    got = {e["locale"]: e for e in list_locales(path)}
    assert got["en_US"] == {"locale": "en_US", "source": "column", "sheet": "Sheet1", "rows": 2, "assigned": None}
    assert got["fr_FR"]["rows"] == 1


def test_a_tab_wins_over_the_column_and_column_only_locales_still_appear(tmp_path):
    path = _write(tmp_path / "wb.xlsx", {"Main sheet": _main(["en_US"] * 4 + ["fr_FR"] * 2), "en_US": _tab(4, 4)})
    got = {e["locale"]: e for e in list_locales(path)}
    assert got["en_US"]["source"] == "tab" and got["fr_FR"]["source"] == "column"
    assert got["fr_FR"]["sheet"] == "Main sheet"


def test_a_workbook_without_locales_lists_none(tmp_path):
    path = _write(tmp_path / "wb.xlsx", {"Sheet1": pd.DataFrame({"rater": ["a"], "justification": ["Fine."]})})
    assert list_locales(path) == []


def test_resolve_picks_the_tab_or_the_column_rows(tmp_path):
    path = _write(tmp_path / "wb.xlsx", {"Main sheet": _main(["en_US"] * 4 + ["fr_FR"] * 2), "en_US": _tab(4, 4)})
    assert resolve(path, "en_US") == ("en_US", None)
    assert resolve(path, "fr_FR") == ("Main sheet", "locale")
    assert resolve(path, "EN-us") == ("en_US", None)  # case and separator don't matter


def test_an_unknown_locale_names_the_ones_there_are(tmp_path):
    path = _write(tmp_path / "wb.xlsx", {"Sheet1": _main(["en_US", "fr_FR"])})
    with pytest.raises(LocaleError, match=r"de_DE.*en_US.*fr_FR"):
        resolve(path, "de_DE")


def test_csv_files_use_their_locale_column(tmp_path):
    path = tmp_path / "rows.csv"
    _main(["ja_JP", "ja_JP", "ko_KR"]).to_csv(path, index=False)
    assert resolve(path, "ja_JP") == ("CSV", "locale")
    assert {e["locale"]: e["rows"] for e in list_locales(path)} == {"ja_JP": 2, "ko_KR": 1}


# --- the pipeline audits one locale -----------------------------------------------------------------------

from src.config import PipelineConfig  # noqa: E402
from src.pipeline import QAPipeline  # noqa: E402


def _pipeline(only_assigned=True):
    config = PipelineConfig()
    config.detector.provider = "heuristic"
    config.structural.skip_unassigned = only_assigned
    return QAPipeline(config=config)


def _mixed(tmp_path):
    main = pd.DataFrame([{"Task Assignee": f"r{i}@x.com", "task_id": f"m{i}", "locale": loc,
                          "Rating_human_rationale": f"The response covers point {i} and the verdict follows."}
                         for i, loc in enumerate(["en_US"] * 4 + ["fr_FR"] * 3)])
    return _write(tmp_path / "wb.xlsx", {"Main sheet": main, "en_US": _tab(4, 2)})


def test_the_pipeline_audits_only_the_chosen_locales_tab(tmp_path):
    res = _pipeline().run(str(_mixed(tmp_path)), str(tmp_path / "out.xlsx"), locale="en_US")
    assert res.summary["locale"] == "en_US" and res.summary["detected_sheet"] == "en_US"
    assert res.total_rows == 2  # the tab's assigned tasks; its padding rows and the other locale never count


def test_the_pipeline_filters_a_locale_column_when_there_is_no_tab(tmp_path):
    res = _pipeline().run(str(_mixed(tmp_path)), str(tmp_path / "out.xlsx"), locale="fr-fr")
    assert res.summary["locale"] == "fr_FR" and res.summary["detected_sheet"] == "Main sheet"
    assert res.total_rows == 3


def test_without_a_locale_the_pipeline_reads_the_sheet_as_before(tmp_path):
    res = _pipeline().run(str(_mixed(tmp_path)), str(tmp_path / "out.xlsx"))
    assert res.summary.get("locale") is None and res.total_rows == 7


def test_the_pipeline_refuses_an_unknown_locale(tmp_path):
    with pytest.raises(LocaleError):
        _pipeline().run(str(_mixed(tmp_path)), str(tmp_path / "out.xlsx"), locale="de_DE")


# --- the API: list locales, inspect one, audit one ---------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _upload(path):
    return {"file": (path.name, path.read_bytes(), XLSX)}


def test_inspect_lists_the_locales_and_inspects_a_chosen_one(tmp_path):
    client, path = TestClient(app), _mixed(tmp_path)
    res = client.post("/api/inspect-columns", files=_upload(path))
    assert res.status_code == 200
    locales = {e["locale"]: e for e in res.json()["locales"]}
    assert locales["en_US"]["source"] == "tab" and locales["fr_FR"]["source"] == "column"
    res = client.post("/api/inspect-columns", files=_upload(path), data={"locale": "en_US"})
    data = res.json()
    assert data["detected_sheet"] == "en_US" and "AR Rationale" in data["recommended"]
    assert data["locale"] == "en_US"
    assert data["locales"] is None  # the page already has the list; re-listing every tab on each pick is slow


def test_audit_runs_one_locale_and_names_it_in_the_output(tmp_path):
    client, path = TestClient(app), _mixed(tmp_path)
    res = client.post("/api/audit", files=_upload(path),
                      data={"provider": "heuristic", "only_assigned": "true", "locale": "fr_FR"})
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows"] == 3 and data["summary"]["locale"] == "fr_FR"
    assert data["output_filename"].endswith("_fr_FR.xlsx")


def test_an_unknown_locale_is_a_clear_400(tmp_path):
    client, path = TestClient(app), _mixed(tmp_path)
    for endpoint, data in (("/api/inspect-columns", {"locale": "de_DE"}),
                           ("/api/audit", {"provider": "heuristic", "locale": "de_DE"})):
        res = client.post(endpoint, files=_upload(path), data=data)
        assert res.status_code == 400 and "de_DE" in res.json()["detail"]


# --- the CLI ----------------------------------------------------------------------------------------------


def test_the_cli_audits_one_locale(tmp_path, capsys):
    import json
    import sys
    from unittest.mock import patch

    from src.cli import main

    path, out = _mixed(tmp_path), tmp_path / "out.xlsx"
    with patch.object(sys, "argv", ["cli.py", "audit", str(path), "-o", str(out), "-p", "heuristic",
                                    "--only-assigned", "--locale", "fr_FR", "--json"]):
        main()
    data = json.loads(capsys.readouterr().out)
    assert data["total_rows"] == 3 and data["summary"]["locale"] == "fr_FR"


def test_the_dashboard_has_a_locale_picker_that_drives_inspect_and_audit():
    page = TestClient(app).get("/").text
    assert 'id="localePanel"' in page and 'id="localeBox"' in page and 'id="localeAll"' not in page  # built on load
    assert "formData.append('locale'" in page and "checkForm.append('locale'" in page
    assert "'All locales'" in page and "for (let i = 0; i < runs.length; i++)" in page   # one audit per locale
    assert 'id="downloadList"' in page and "mergeSummaries(" in page


def test_the_dashboard_names_the_served_model():
    page = TestClient(app).get("/").text
    assert 'id="modelBadge"' in page and "fetch('/api/v2/model')" in page
    info = TestClient(app).get("/api/v2/model").json()
    assert set(info) == {"model_id", "validated"}


def test_inspecting_a_workbook_leaves_no_upload_behind(tmp_path):
    """The temp upload is deleted after an inspect. On Windows an unclosed ExcelFile kept it locked, so the delete
    failed silently and uploaded rater workbooks piled up in the uploads folder."""
    import os

    from src.api import UPLOAD_DIR

    client, path = TestClient(app), _mixed(tmp_path)
    before = {f for f in os.listdir(UPLOAD_DIR) if f.startswith("inspect_")}
    for data in ({}, {"locale": "en_US"}, {"locale": "fr_FR"}):
        assert client.post("/api/inspect-columns", files=_upload(path), data=data).status_code == 200
    assert {f for f in os.listdir(UPLOAD_DIR) if f.startswith("inspect_")} == before


def test_dialect_column_with_variant_codes_lists_and_resolves(tmp_path):
    """A `dialect` column doubles as the locale; codes may carry a variant (hi_in_bhojpuri, ja_jp_kansai)."""
    df = pd.DataFrame({"Task ID": [f"t{i}" for i in range(5)],
                       "dialect": ["en_au", "en_au", "hi_in_bhojpuri", "ja_jp_kansai", "fr_ca"],
                       "feedback": ["Some feedback text for this row."] * 5})
    path = _write(tmp_path / "d.xlsx", {"Export": df})
    found = {e["locale"]: e["rows"] for e in list_locales(path)}
    assert found == {"en_au": 2, "hi_in_bhojpuri": 1, "ja_jp_kansai": 1, "fr_ca": 1}
    assert resolve(path, "HI_IN_Bhojpuri") == ("Export", "dialect")
