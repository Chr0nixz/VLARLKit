#!/usr/bin/env bash
# Install the RoboTwin benchmark in a uv environment inside the RoboTwin checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROBOTWIN_SRC="${ROBOTWIN_DIR:-$SCRIPT_DIR/RoboTwin}"
ROBOTWIN_VENV="${ROBOTWIN_VENV:-$ROBOTWIN_SRC/.venv}"
PYTHON_BIN="$ROBOTWIN_VENV/bin/python"
ROBOTWIN_REF="${ROBOTWIN_REF:-RLinf_support}"
TORCH_BACKEND="${ROBOTWIN_TORCH_BACKEND:-cu121}"
CUROBO_REPO="${CUROBO_REPO:-https://github.com/NVlabs/curobo.git}"
CUROBO_REF="${CUROBO_REF:-v0.7.8}"
DOWNLOAD_ASSETS=1

usage() {
    cat <<EOF
Usage: bash third_party/install_robotwin.sh [--skip-assets]

Environment variables:
  ROBOTWIN_DIR            RoboTwin source directory. Default: third_party/RoboTwin
  ROBOTWIN_VENV           uv environment path. Default: third_party/RoboTwin/.venv
  ROBOTWIN_REF            RoboTwin git ref to install. Default: RLinf_support
  ROBOTWIN_TORCH_BACKEND  uv torch backend. Default: cu121
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-assets)
            DOWNLOAD_ASSETS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

ensure_robotwin_source() {
    if [[ -d "$ROBOTWIN_SRC/.git" ]]; then
        echo "Using existing RoboTwin source at $ROBOTWIN_SRC"
        return
    fi

    if [[ "$ROBOTWIN_SRC" == "$SCRIPT_DIR/RoboTwin" ]]; then
        echo "Initializing RoboTwin submodule at $ROBOTWIN_SRC"
        git -C "$REPO_ROOT" submodule update --init --recursive third_party/RoboTwin
        return
    fi

    if [[ -e "$ROBOTWIN_SRC" ]]; then
        echo "$ROBOTWIN_SRC exists but is not a git checkout. Please move it aside or set ROBOTWIN_DIR." >&2
        exit 1
    fi

    echo "ROBOTWIN_DIR must point to an existing RoboTwin git checkout when overriding the default path." >&2
    exit 1
}

checkout_robotwin_ref() {
    local status_output
    status_output="$(git -C "$ROBOTWIN_SRC" status --porcelain)"
    if [[ -n "$status_output" ]]; then
        echo "$ROBOTWIN_SRC has uncommitted changes. Commit, stash, or clean them before installing $ROBOTWIN_REF." >&2
        echo "$status_output" >&2
        exit 1
    fi

    echo "Checking out RoboTwin ref: $ROBOTWIN_REF"
    git -C "$ROBOTWIN_SRC" fetch origin "$ROBOTWIN_REF"
    git -C "$ROBOTWIN_SRC" checkout -B "$ROBOTWIN_REF" "origin/$ROBOTWIN_REF"
    git -C "$ROBOTWIN_SRC" submodule update --init --recursive
}

create_venv() {
    echo "Creating isolated uv environment at $ROBOTWIN_VENV"
    uv venv --no-project --allow-existing --python 3.10 "$ROBOTWIN_VENV"
}

install_robotwin_source_path() {
    local site_packages
    local robotwin_src_abs

    robotwin_src_abs="$(cd "$ROBOTWIN_SRC" && pwd)"
    site_packages="$("$PYTHON_BIN" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"

    mkdir -p "$site_packages"
    printf '%s\n' "$robotwin_src_abs" > "$site_packages/robotwin_source.pth"
    echo "Installed RoboTwin source path into $site_packages/robotwin_source.pth"
}

uv_install() {
    uv pip install --python "$PYTHON_BIN" --torch-backend "$TORCH_BACKEND" "$@"
}

uv_show_location() {
    uv pip show --python "$PYTHON_BIN" "$1" | awk '/^Location:/ {print $2}'
}

patch_sapien() {
    local location
    location="$(uv_show_location sapien)"
    local urdf_loader="$location/sapien/wrapper/urdf_loader.py"

    if [[ ! -f "$urdf_loader" ]]; then
        echo "Cannot find sapien urdf_loader.py at $urdf_loader" >&2
        exit 1
    fi

    echo "Patching sapien wrapper/urdf_loader.py"
    "$PYTHON_BIN" - "$urdf_loader" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
patched = re.sub(r'("r")(\))(\s+as)', r'\1, encoding="utf-8")\3', text)
if patched != text:
    path.write_text(patched)
PY
}

patch_mplib() {
    local location
    location="$(uv_show_location mplib)"
    local planner="$location/mplib/planner.py"

    if [[ ! -f "$planner" ]]; then
        echo "Cannot find mplib planner.py at $planner" >&2
        exit 1
    fi

    echo "Patching mplib planner.py"
    "$PYTHON_BIN" - "$planner" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
new = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit(f"Could not find expected mplib planner condition in {path}")
PY
}

install_curobo() {
    local curobo_dir="$ROBOTWIN_SRC/envs/curobo"

    if [[ -d "$curobo_dir/.git" ]]; then
        echo "Using existing CuRobo source at $curobo_dir"
    else
        if [[ -e "$curobo_dir" ]]; then
            echo "$curobo_dir exists but is not a git checkout. Please move it aside." >&2
            exit 1
        fi

        echo "Cloning CuRobo ($CUROBO_REF) into $curobo_dir"
        git clone --branch "$CUROBO_REF" --depth 1 "$CUROBO_REPO" "$curobo_dir"
    fi

    echo "Installing CuRobo"
    uv_install -e "$curobo_dir" --no-build-isolation
}

download_assets() {
    if [[ "$DOWNLOAD_ASSETS" -eq 0 ]]; then
        echo "Skipping RoboTwin asset download."
        return
    fi

    echo "Downloading RoboTwin assets"
    (
        cd "$ROBOTWIN_SRC"
        PATH="$ROBOTWIN_VENV/bin:$PATH" bash script/_download_assets.sh
    )
}

require_command git
require_command uv
require_command unzip

ensure_robotwin_source
checkout_robotwin_ref
create_venv
install_robotwin_source_path

echo "Installing RoboTwin requirements"
uv_install -r "$ROBOTWIN_SRC/script/requirements.txt"

echo "Installing pytorch3d"
uv_install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

patch_sapien
patch_mplib
install_curobo

echo "Installing RoboTwin post-requirements"
uv_install "warp-lang==1.12.0" "setuptools==69.5.1"

download_assets

cat <<EOF
RoboTwin installation complete.

Environment:
  source "$ROBOTWIN_VENV/bin/activate"

Source:
  cd "$ROBOTWIN_SRC"
EOF
