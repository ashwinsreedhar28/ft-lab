#!/usr/bin/env bash
# Fetch the Spider SQLite databases into data/spider/database/<db_id>/<db_id>.sqlite
# Source: HAL-9001/spider-databases on the HF Hub (re-host of the official spider_data.zip).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/spider
EXPECTED=00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b
if [ ! -f data/spider/spider_data.zip ]; then
  python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('HAL-9001/spider-databases', 'spider_data.zip', repo_type='dataset', local_dir='data/spider')
print(p)"
fi
echo "$EXPECTED  data/spider/spider_data.zip" | sha256sum -c -
python - <<'EOF'
import zipfile, pathlib
z = zipfile.ZipFile("data/spider/spider_data.zip")
names = [n for n in z.namelist() if "/database/" in n and n.endswith(".sqlite")]
root = pathlib.Path("data/spider/database")
for n in names:
    db = n.split("/database/")[1]            # <db_id>/<db_id>.sqlite
    dest = root / db
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(z.read(n))
print(f"extracted {len(names)} databases to {root}")
EOF
