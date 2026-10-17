import sqlite3

import pytest

from ftlab.config import load_config
from ftlab.data import gsm8k, spider
from ftlab.rewards import get_rewards
from ftlab.rewards import gsm8k as rg
from ftlab.rewards import sql as rs


# ---------- GSM8K ----------
def test_gold_answer_parsing():
    assert gsm8k.gold_answer("blah\n#### 1,234") == "1234"
    assert gsm8k.gold_answer("x #### 3.50") == "3.5"
    assert gsm8k.gold_answer("x #### -7") == "-7"


@pytest.mark.parametrize("text,expect", [
    ("so the answer is\n#### 42", "42"),
    ("#### $1,200", "1200"),
    ("I think 17 then 18.\n#### 18", "18"),
    ("no marker but 5 apples and 3 pears = 8", "8"),  # lenient fallback
    ("nothing numeric here", None),
])
def test_extract_pred(text, expect):
    assert gsm8k.extract_pred(text) == expect


def test_extract_strict_requires_marker():
    assert gsm8k.extract_pred("answer 8", strict=True) is None


def test_correct_reward_shapes_and_values():
    comps = [[{"role": "assistant", "content": "#### 8"}], [{"role": "assistant", "content": "#### 9"}], "#### 8"]
    r = rg.correct(comps, answer=["8", "8", "8"])
    assert r == [1.0, 0.0, 1.0]
    assert rg.format_ok(["#### 1", "one"]) == [0.2, 0.0]


def test_correct_reward_logs_metric():
    seen = {}
    rg.correct(["#### 1", "#### 2"], answer=["1", "1"], log_metric=lambda k, v: seen.__setitem__(k, v))
    assert seen["gsm8k/acc"] == 0.5


def test_length_penalty():
    assert rg.length_penalty([[0] * 512, [0] * 768]) == [0.0, pytest.approx(-0.1)]


# ---------- Spider ----------
@pytest.fixture
def db(tmp_path):
    root = tmp_path / "database"
    d = root / "toy"
    d.mkdir(parents=True)
    con = sqlite3.connect(d / "toy.sqlite")
    con.executescript(
        "CREATE TABLE emp(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL);"
        "INSERT INTO emp VALUES (1,'a','x',10),(2,'b','x',20),(3,'c','y',30);"
    )
    con.commit(); con.close()
    return root


def test_schema_text(db):
    s = spider.schema_text("toy", str(db))
    assert s.startswith("CREATE TABLE emp") and s.endswith(";")


@pytest.mark.parametrize("text,expect", [
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
    ("thinking...\n```sql\nSELECT a\nFROM b\n```\n", "SELECT a\nFROM b"),
    ("```\nSELECT 2\n```", "SELECT 2"),
    ("Here: SELECT name FROM emp WHERE id = 1", "SELECT name FROM emp WHERE id = 1"),
    ("no query", None),
])
def test_extract_sql(text, expect):
    assert spider.extract_sql(text) == expect


def test_exec_match_set_vs_order(db):
    gold = "SELECT name FROM emp WHERE dept='x'"
    assert spider.exec_match("SELECT name FROM emp WHERE dept='x' ORDER BY name DESC", gold, "toy", db)[0]
    gold_ordered = "SELECT name FROM emp ORDER BY salary DESC"
    ok, _ = spider.exec_match("SELECT name FROM emp ORDER BY salary ASC", gold_ordered, "toy", db)
    assert not ok


def test_exec_match_errors(db):
    ok, err = spider.exec_match("SELECT nope FROM emp", "SELECT 1", "toy", db)
    assert not ok and "operational" in err
    ok, err = spider.exec_match(None, "SELECT 1", "toy", db)
    assert not ok and err == "no sql"


def test_exec_is_read_only(db):
    rows, err = spider.execute("DELETE FROM emp", "toy", db)
    assert rows is None and err


def test_sql_rewards(db):
    comps = ["```sql\nSELECT count(*) FROM emp\n```", "```sql\nSELECT count(*) FROM emp WHERE 1=0\n```", "SELECT bad"]
    r = rs.exec_match(comps, gold_sql=["SELECT 3"] * 3, db_id=["toy"] * 3, db_dir=[str(db)] * 3)
    assert r == [1.0, 0.0, 0.0]
    assert rs.runs(comps, gold_sql=["SELECT 3"] * 3, db_id=["toy"] * 3, db_dir=[str(db)] * 3) == [0.1, 0.1, 0.0]


def test_score_matches_reward(db):
    row = {"gold_sql": "SELECT 3", "db_id": "toy", "db_dir": str(db)}
    assert spider.score("```sql\nSELECT count(*) FROM emp\n```", row) == 1.0


# ---------- config ----------
def test_config_extends_chain_and_overrides():
    cfg = load_config("configs/gsm8k_smoke.yaml", ["grpo.max_steps=7", "data.train_limit=5"])
    assert cfg.name == "gsm8k_smoke"
    assert cfg.grpo["max_steps"] == 7 and cfg.data.train_limit == 5
    assert cfg.grpo["beta"] == 0.04            # inherited two levels up from base.yaml
    assert cfg.grpo["max_completion_length"] == 256
    assert str(cfg.resolved_output_dir()) == "runs/gsm8k_smoke"


def test_all_configs_load_and_rewards_resolve():
    import glob
    for p in glob.glob("configs/*.yaml"):
        cfg = load_config(p)
        get_rewards(cfg.rewards)
