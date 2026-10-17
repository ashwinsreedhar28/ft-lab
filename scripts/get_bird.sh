#!/usr/bin/env bash
# BIRD dev: 1,534 questions, 11 SQLite databases (~1.7 GB). Pod only; too big for the Mac clone.
# Layout after: data/bird/dev.json, data/bird/database/<db_id>/<db_id>.sqlite
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/bird && cd data/bird
[ -f dev.zip ] || curl -L -o dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
python - <<'PY'
import zipfile, pathlib, shutil, io
root = pathlib.Path(".")
z = zipfile.ZipFile("dev.zip")
z.extractall("unz")
# dev.json
js = next(root.glob("unz/**/dev.json"))
shutil.copy(js, "dev.json")
# databases: either a nested dev_databases.zip or an extracted dev_databases/ tree
inner = list(root.glob("unz/**/dev_databases.zip"))
if inner:
    zipfile.ZipFile(inner[0]).extractall("unz/dbs")
dbdir = pathlib.Path("database"); dbdir.mkdir(exist_ok=True)
n = 0
for sq in root.glob("unz/**/*.sqlite"):
    dest = dbdir / sq.parent.name / sq.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy(sq, dest)
    n += 1
shutil.rmtree("unz")
print(f"dev.json + {n} sqlite files -> {dbdir}")
PY
