#!/bin/bash
# Public, isolated Xing4_0 installer. Never changes FlagOS/current or shell files.
set -euo pipefail
VERSION=0.2.0-alpha.2-xing4-0
ARCHIVE=flagos-runtime-$VERSION-darwin-arm64-m5pro.tar.gz
BASE=https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v$VERSION
ASSET=
DEST=./runtime
ACTION=install
while [ "$#" -gt 0 ]; do
  case "$1" in
    --asset) [ "$#" -ge 2 ] || exit 2; ASSET=$2; shift 2 ;;
    --destination) [ "$#" -ge 2 ] || exit 2; DEST=$2; shift 2 ;;
    --uninstall) [ "$#" -ge 2 ] || exit 2; ACTION=uninstall; DEST=$2; shift 2 ;;
    -h|--help)
      echo 'Usage: bash install.sh [--asset ARCHIVE] [--destination DIRECTORY]'
      echo '       bash install.sh --uninstall DIRECTORY'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ "$ACTION" = uninstall ]; then
  [ -d "$DEST" ] && [ ! -L "$DEST" ] || { echo 'Select a real installed directory, not a symlink.' >&2; exit 2; }
  PHYSICAL_DEST=$(CDPATH= cd -- "$DEST" && pwd -P)
  [ -f "$PHYSICAL_DEST/share/flagos/install-receipt.json" ] || {
    echo 'Missing managed-install receipt; refusing to remove this directory.' >&2; exit 2;
  }
  "$PHYSICAL_DEST/bin/vllm" --flagos-run-python - "$PHYSICAL_DEST" "$VERSION" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
receipt = json.loads((root / 'share/flagos/install-receipt.json').read_text())
manifest = json.loads((root / 'share/flagos/runtime-manifest.json').read_text())
if (receipt.get('version') != sys.argv[2] or manifest.get('version') != sys.argv[2]
        or receipt.get('destination') != str(root) or receipt.get('managed') is not True):
    raise SystemExit('Receipt/version mismatch; refusing to remove this directory.')
PY
  TRASH_ROOT="$HOME/.Trash"
  mkdir -p "$TRASH_ROOT"
  TRASH_TARGET="$TRASH_ROOT/FlagOS-$VERSION-$(date +%Y%m%d-%H%M%S)-$$"
  [ ! -e "$TRASH_TARGET" ] || exit 2
  mv "$PHYSICAL_DEST" "$TRASH_TARGET"
  echo "Uninstalled; recoverable in Trash: $TRASH_TARGET"
  exit 0
fi
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || {
  echo 'Native macOS arm64 is required.' >&2; exit 2;
}
[ "$(/usr/sbin/sysctl -n hw.model)" = Mac17,9 ] || {
  echo 'This release is validated on Mac17,9 / Apple M5 Pro.' >&2; exit 2;
}
for feature in DotProd I8MM; do
  [ "$(/usr/sbin/sysctl -n "hw.optional.arm.FEAT_$feature")" = 1 ] || {
    echo "Missing Arm feature: $feature" >&2; exit 2;
  }
done
case "$DEST" in *[[:space:]]*) echo 'Use a destination without whitespace.' >&2; exit 2 ;; esac
[ ! -e "$DEST" ] && [ ! -L "$DEST" ] || {
  echo 'Destination already exists; refusing to overwrite it.' >&2; exit 2;
}
if [ -z "$ASSET" ]; then
  DOWNLOAD=$(mktemp -d ./.flagos-download.XXXXXX)
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 1800 \
    "$BASE/$ARCHIVE" --output "$DOWNLOAD/$ARCHIVE"
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 120 \
    "$BASE/$ARCHIVE.sha256" --output "$DOWNLOAD/$ARCHIVE.sha256"
  ASSET="$DOWNLOAD/$ARCHIVE"
fi
[ -f "$ASSET" ] && [ -f "$ASSET.sha256" ] || {
  echo 'Archive and matching SHA256 sidecar are required.' >&2; exit 2;
}
ASSET_DIR=$(CDPATH= cd -- "$(dirname -- "$ASSET")" && pwd)
[ "$(basename -- "$ASSET")" = "$ARCHIVE" ] || {
  echo "Expected archive: $ARCHIVE" >&2; exit 2;
}
(cd "$ASSET_DIR" && shasum -a 256 -c "$ARCHIVE.sha256")
EXPECTED_ROOT=flagos-runtime-$VERSION-darwin-arm64-m5pro
while IFS= read -r member; do
  case "$member" in *'/../'*|*'/./'*) echo 'Unsafe archive path.' >&2; exit 2 ;; esac
  case "$member" in
    "$EXPECTED_ROOT"|"$EXPECTED_ROOT/"|"$EXPECTED_ROOT/"*) ;;
    *) echo 'Unexpected archive layout.' >&2; exit 2 ;;
  esac
done < <(tar -tzf "$ASSET_DIR/$ARCHIVE")
mkdir -p "$(dirname -- "$DEST")"
STAGING=$(mktemp -d "$(dirname -- "$DEST")/.flagos-install.XXXXXX")
tar -xzf "$ASSET_DIR/$ARCHIVE" -C "$STAGING"
EXTRACTED="$STAGING/$EXPECTED_ROOT"
[ -x "$EXTRACTED/bin/vllm" ] || { echo 'Runtime launcher is missing.' >&2; exit 2; }
"$EXTRACTED/bin/vllm" --flagos-run-python \
  "$EXTRACTED/share/flagos/verify_runtime.py" "$EXTRACTED"
"$EXTRACTED/bin/vllm" --version
mv "$EXTRACTED" "$DEST"
rmdir "$STAGING"
"$DEST/bin/vllm" --flagos-run-python - "$DEST" "$VERSION" "$ARCHIVE" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
receipt = {'version': sys.argv[2], 'archive': sys.argv[3], 'destination': str(root), 'managed': True}
(root / 'share/flagos/install-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
PY
"$DEST/bin/vllm" --version
echo "Installed isolated Runtime: $DEST"
echo 'Use ./runtime/bin/vllm (or the destination you selected).'
