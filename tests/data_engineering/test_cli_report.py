from collections import deque

import pandas as pd
import pytest

from src.data import cli
from src.data.report import md_table, render
from src.db.queries import detected_constructs


def _scripted(*answers):
    queue = deque(answers)

    def fake_input(_prompt=""):
        if not queue:
            raise EOFError
        return queue.popleft()

    return fake_input


# ----------------------------------------------------------------------------- menu
def test_menu_exits_on_zero(capsys):
    assert cli.run_menu(_scripted("0")) == 0
    assert "Bye." in capsys.readouterr().out


def test_menu_survives_invalid_input_and_reports_known_errors(capsys, monkeypatch):
    def broken_step():
        raise FileNotFoundError("data/raw/creditcard.csv not found")

    monkeypatch.setitem(cli.COMMANDS, "audit", ("Audit", broken_step))
    audit_number = list(cli.COMMANDS).index("audit") + 1
    status = cli.run_menu(_scripted("abc", "999", "-3", str(audit_number), "0"))
    out = capsys.readouterr().out
    assert status == 0
    assert "'abc' is not a number" in out and "does not exist" in out
    assert "Problem: data/raw/creditcard.csv not found" in out  # menu keeps running after the error


def test_menu_ends_cleanly_when_input_is_closed():
    assert cli.run_menu(_scripted()) == 0  # immediate EOF


def test_every_command_is_listed_in_the_parser_and_the_menu(capsys):
    parser_text = cli.build_parser().format_help()
    for name in cli.COMMANDS:
        assert name in parser_text
    cli.run_menu(_scripted("0"))
    menu_text = capsys.readouterr().out
    for _name, (description, _func) in cli.COMMANDS.items():
        assert description in menu_text


# ----------------------------------------------------------------------------- arguments
def test_parse_params():
    assert cli.parse_params(["min_transactions=500", "split_name=train", "ratio=0.5"]) == {
        "min_transactions": 500, "split_name": "train", "ratio": 0.5}
    assert cli.parse_params(None) == {}
    for bad in (["novalue"], ["=3"]):
        with pytest.raises(ValueError, match="name=value"):
            cli.parse_params(bad)


def test_main_returns_error_code_for_missing_data(monkeypatch, tmp_path, capsys):
    import src.db.engine as engine_module
    import src.db.queries as queries_module

    missing = tmp_path / "nope.db"
    monkeypatch.setattr(queries_module, "get_engine", lambda *a, **k: engine_module.get_engine(missing))
    assert cli.main(["query", "q01_split_overview"]) == 1
    assert "build-db" in capsys.readouterr().err


def test_main_lists_queries_without_a_database(capsys):
    assert cli.main(["query"]) == 0
    assert "q01_split_overview" in capsys.readouterr().out


# ----------------------------------------------------------------------------- report helpers
def test_md_table_shape():
    table = md_table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |", "| 2 | y |"]


def test_render_fails_loudly_on_a_missing_placeholder(tmp_path, monkeypatch):
    import src.data.report as report

    (tmp_path / "t.tmpl").write_text("value: ${present} and ${absent}")
    monkeypatch.setattr(report, "TEMPLATES_DIR", tmp_path)
    with pytest.raises(KeyError, match="absent"):
        render("t.tmpl", {"present": "1"})
    assert render("t.tmpl", {"present": "1", "absent": "2"}) == "value: 1 and 2"


def test_subquery_detector_ignores_cte_bodies():
    cte_only = "WITH a AS (\n SELECT 1)\nSELECT * FROM a"
    nested = "SELECT * FROM t WHERE x > (SELECT AVG(x) FROM t)"
    exists = "SELECT 1 WHERE NOT EXISTS (\n SELECT 1)"
    assert "subquery" not in detected_constructs(cte_only) and "CTE (WITH)" in detected_constructs(cte_only)
    assert "subquery" in detected_constructs(nested)
    assert "subquery" in detected_constructs(exists)
