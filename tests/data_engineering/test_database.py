import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from src.config import AMOUNT_BANDS, SPLIT_NAMES, V_COLUMNS
from src.data.storage import save_parquet
from src.db.builder import build_database, verify_database
from src.db.engine import get_engine
from src.db.er_diagram import generate_er_diagram, read_model, to_mermaid


@pytest.fixture
def database(tmp_path, processed_frame):
    parquet = save_parquet(processed_frame, tmp_path / "transactions.parquet")
    db_path = tmp_path / "fraud.db"
    report = build_database(parquet, db_path)
    return db_path, report


def test_build_creates_schema_and_verified_content(database, processed_frame):
    db_path, report = database
    assert report["ok"] and report["foreign_key_violations"] == 0 and report["integrity_check"] == "ok"
    engine = get_engine(db_path)
    tables = set(inspect(engine).get_table_names())
    assert tables == {"splits", "classes", "time_slots", "amount_bands", "transactions", "transaction_features"}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar_one() == len(processed_frame)
        assert conn.execute(text("SELECT COUNT(*) FROM splits")).scalar_one() == len(SPLIT_NAMES)
        assert conn.execute(text("SELECT COUNT(*) FROM amount_bands")).scalar_one() == len(AMOUNT_BANDS)
        columns = [c["name"] for c in inspect(conn).get_columns("transaction_features")]
        assert columns == ["transaction_id", *[c.lower() for c in V_COLUMNS]]
        hours = conn.execute(text("SELECT MAX(hour_index) FROM time_slots")).scalar_one()
        assert hours == int(processed_frame["Time"].max()) // 3600


def test_constraints_reject_bad_rows(database):
    engine = get_engine(database[0])
    bad_statements = [
        "INSERT INTO transactions VALUES (999999, 99, 0, 0, 1, 1.0)",  # unknown split_id
        "INSERT INTO transactions VALUES (999999, 1, 5, 0, 1, 1.0)",  # class outside 0/1
        "INSERT INTO transactions VALUES (999999, 1, 0, 0, 1, -5.0)",  # negative amount
        "INSERT INTO transactions VALUES (0, 1, 0, 0, 1, 1.0)",  # duplicate primary key
        "INSERT INTO splits VALUES (10, 'train', 'duplicate name')",  # UNIQUE split_name
        "INSERT INTO transaction_features (transaction_id, v1) VALUES (12345678, 0.0)",  # FK + NOT NULL
    ]
    for statement in bad_statements:
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                conn.execute(text(statement))


def test_cascade_delete_removes_features(database):
    engine = get_engine(database[0])
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM transactions WHERE transaction_id = 0"))
        left = conn.execute(text("SELECT COUNT(*) FROM transaction_features WHERE transaction_id = 0")).scalar_one()
    assert left == 0


def test_verify_detects_tampering(database, processed_frame):
    engine = get_engine(database[0])
    with engine.begin() as conn:
        conn.execute(text("UPDATE transactions SET amount = amount + 1 WHERE transaction_id = 5"))
    with pytest.raises(ValueError, match="amount sums differ"):
        verify_database(engine, processed_frame)


def test_failed_build_leaves_no_files(tmp_path, processed_frame):
    broken = processed_frame.copy()
    broken["Amount"] = -1.0  # violates CHECK (amount >= 0)
    parquet = save_parquet(broken, tmp_path / "bad.parquet")
    db_path = tmp_path / "fraud.db"
    with pytest.raises(Exception):
        build_database(parquet, db_path)
    assert not db_path.exists() and not list(tmp_path.glob("*.tmp"))


def test_overwrite_flag_and_missing_file_errors(database, tmp_path):
    db_path, _ = database
    parquet = tmp_path / "transactions.parquet"
    with pytest.raises(FileExistsError):
        build_database(parquet, db_path, overwrite=False)
    with pytest.raises(FileNotFoundError, match="build-db"):
        get_engine(tmp_path / "nope.db")


def test_er_model_is_read_from_the_database(database, tmp_path):
    engine = get_engine(database[0])
    tables, relations = read_model(engine)
    pairs = {(r.child, r.parent, r.one_to_one, r.logical) for r in relations}
    assert ("transactions", "splits", False, False) in pairs
    assert ("transaction_features", "transactions", True, False) in pairs
    assert ("transactions", "amount_bands", False, True) in pairs
    mermaid = to_mermaid(tables, relations)
    assert mermaid.startswith("erDiagram") and "transactions {" in mermaid and "v1_to_v28" in mermaid

    outputs = generate_er_diagram(database[0], tmp_path / "docs")
    assert outputs["png"].stat().st_size > 10_000
    assert "```mermaid" in outputs["markdown"].read_text()
