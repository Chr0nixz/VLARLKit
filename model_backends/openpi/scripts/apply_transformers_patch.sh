#!/bin/bash
set -euo pipefail

python - <<'PY'
import pathlib
import shutil
import site

site_packages = pathlib.Path(site.getsitepackages()[0])
src = site_packages / "openpi" / "models_pytorch" / "transformers_replace"
dst = site_packages / "transformers"

if not src.exists():
    raise SystemExit(f"OpenPI transformers patch source not found: {src}")
if not dst.exists():
    raise SystemExit(f"transformers package not found: {dst}")

for item in src.iterdir():
    target = dst / item.name
    if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        shutil.copy2(item, target)

print(f"Applied OpenPI transformers patch from {src} to {dst}")
PY
