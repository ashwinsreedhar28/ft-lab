import subprocess
from pathlib import Path

from review.agent.loop import container_kw, prelude
from review.mine.assemble import mutant_commit


def test_mutant_commit_is_pr_tree_plus_mutation(tmp_path: Path):
    repo = tmp_path / "r"; repo.mkdir()
    def g(*a, **k): return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True, **k).stdout.strip()
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (repo / "m.py").write_text("x = 1\n"); g("add", "."); g("commit", "-qm", "base")
    parent = g("rev-parse", "HEAD")
    (repo / "m.py").write_text("x = 1\ny = 2\n"); (repo / "n.py").write_text("z = 3\n"); g("add", "."); g("commit", "-qm", "pr")
    pr = g("rev-parse", "HEAD")
    sha = mutant_commit(repo, pr, parent, "m.py", "x = 1\ny = 3\n", "mutant")
    assert g("rev-parse", f"{sha}^") == parent
    assert g("show", f"{sha}:m.py") == "x = 1\ny = 3" and g("show", f"{sha}:n.py") == "z = 3"
    assert "+y = 3" in g("show", "--format=", sha) and "+z = 3" in g("show", "--format=", sha)
    assert g("status", "--porcelain") == ""          # working tree and real index untouched
    assert mutant_commit(repo, pr, parent, "m.py", "x = 1\ny = 3\n", "mutant") == sha   # deterministic


def test_prelude_and_mount_for_mined_tasks():
    shwin = {"image": "x"}
    mined = {"image": "rb-mine-click", "setup_cmd": "git clone -q /repo /testbed; cd /testbed", "repo_mount": "/c", "network": "bridge"}
    assert prelude(shwin, "abc") == "cd /testbed && git checkout -q abc"
    assert prelude(mined, "abc").endswith("cd /testbed && git checkout -q abc")
    assert container_kw(shwin) == {"network": "none"}
    assert container_kw(mined) == {"network": "bridge", "mounts": ["/c:/repo:ro"]}
    assert container_kw({"image": "x"}, network="bridge")["network"] == "bridge"
