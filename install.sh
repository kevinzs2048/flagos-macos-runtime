#!/bin/bash
set -euo pipefail

VERSION=0.1.0-alpha.2
APP_ROOT=${FLAGOS_INSTALL_ROOT:-"$HOME/Library/FlagOS"}
ASSET=
ACTION=install
TARGET_VERSION=
RELEASE_BASE=${FLAGOS_RELEASE_BASE_URL:-"https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v$VERSION"}

usage() {
  echo "Usage: install.sh [--asset FILE] [--version VERSION]"
  echo "       install.sh --rollback VERSION"
  echo "       install.sh --uninstall VERSION"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --asset) ASSET=$2; shift 2 ;;
    --version) VERSION=$2; shift 2 ;;
    --rollback) ACTION=rollback; TARGET_VERSION=$2; shift 2 ;;
    --uninstall) ACTION=uninstall; TARGET_VERSION=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

validate_version() {
  case "$1" in
    ""|.|..|*/*|*[^A-Za-z0-9._-]*)
      echo "Invalid Runtime version: $1" >&2
      exit 2
      ;;
  esac
}

if [ "$ACTION" = install ]; then
  case "$VERSION" in v*) VERSION=${VERSION#v} ;; esac
  validate_version "$VERSION"
  if [ -z "${FLAGOS_RELEASE_BASE_URL:-}" ]; then
    RELEASE_BASE="https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v$VERSION"
  fi
else
  case "$TARGET_VERSION" in v*) TARGET_VERSION=${TARGET_VERSION#v} ;; esac
  validate_version "$TARGET_VERSION"
fi

case "$APP_ROOT" in
  *[[:space:]]*)
    echo "Runtime install path must not contain whitespace: $APP_ROOT" >&2
    echo "PyTorch Inductor cannot compile CPU sampler library paths containing whitespace." >&2
    exit 2
    ;;
esac

RUNTIMES="$APP_ROOT/runtimes"
CURRENT="$APP_ROOT/current"
mkdir -p "$RUNTIMES" "$APP_ROOT/cache"

activate_runtime() {
  selected=$1
  [ -x "$selected/bin/vllm" ] || {
    echo "Runtime is incomplete: $selected" >&2
    exit 2
  }
  current_link="$APP_ROOT/.current.$$"
  /bin/ln -s "$selected" "$current_link"
  /bin/mv -fh "$current_link" "$CURRENT"
}

if [ "$ACTION" = rollback ]; then
  destination="$RUNTIMES/$TARGET_VERSION"
  [ -d "$destination" ] || { echo "Runtime is not installed: $TARGET_VERSION" >&2; exit 2; }
  activate_runtime "$destination"
  echo "FlagOS current Runtime -> $TARGET_VERSION"
  exit 0
fi

if [ "$ACTION" = uninstall ]; then
  destination="$RUNTIMES/$TARGET_VERSION"
  [ -d "$destination" ] || { echo "Runtime is not installed: $TARGET_VERSION" >&2; exit 2; }
  if [ -L "$CURRENT" ] && [ "$(readlink "$CURRENT")" = "$destination" ]; then
    echo "Refusing to uninstall the active Runtime; roll back first." >&2
    exit 2
  fi
  /bin/rm -rf -- "$destination"
  echo "Removed $destination"
  exit 0
fi

[ "$(/usr/bin/uname -m)" = arm64 ] || {
  echo "This Runtime requires Apple arm64." >&2
  exit 2
}
HW_MODEL=$(/usr/sbin/sysctl -n hw.model 2>/dev/null || true)
[ "$HW_MODEL" = Mac17,9 ] || {
  echo "This release is validated only for Mac17,9 / Apple M5 Pro; found ${HW_MODEL:-unknown}." >&2
  exit 2
}
for feature in DotProd I8MM; do
  [ "$(/usr/sbin/sysctl -n "hw.optional.arm.FEAT_$feature" 2>/dev/null || true)" = 1 ] || {
    echo "Required Arm feature is missing: FEAT_$feature" >&2
    exit 2
  }
done

download_dir=
if [ -z "$ASSET" ]; then
  download_dir=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/flagos-download.XXXXXX")
  trap '/bin/rm -rf -- "$download_dir"' EXIT
  archive="flagos-runtime-$VERSION-darwin-arm64-m5pro.tar.gz"
  parts_manifest="$archive.parts"
  CURL_ARGS=(--fail --location --silent --show-error --retry 3 --connect-timeout 20 --max-time 1800)
  /usr/bin/curl "${CURL_ARGS[@]}" "$RELEASE_BASE/$archive.sha256" -o "$download_dir/$archive.sha256"
  /usr/bin/curl "${CURL_ARGS[@]}" "$RELEASE_BASE/$parts_manifest" -o "$download_dir/$parts_manifest"

  parts=()
  while read -r digest part extra; do
    [ -z "${extra:-}" ] || { echo "Invalid Runtime part manifest entry" >&2; exit 2; }
    case "$digest" in
      ""|*[!0-9A-Fa-f]*) echo "Invalid Runtime part digest" >&2; exit 2 ;;
    esac
    [ "${#digest}" -eq 64 ] || { echo "Invalid Runtime part digest" >&2; exit 2; }
    expected_part=$(/usr/bin/printf '%s.part-%03d' "$archive" "${#parts[@]}")
    [ "$part" = "$expected_part" ] || {
      echo "Unexpected Runtime part: $part" >&2
      exit 2
    }
    parts+=("$part")
  done < "$download_dir/$parts_manifest"
  [ "${#parts[@]}" -gt 0 ] || { echo "Runtime part manifest is empty" >&2; exit 2; }

  pids=()
  for part in "${parts[@]}"; do
    /usr/bin/curl "${CURL_ARGS[@]}" "$RELEASE_BASE/$part" -o "$download_dir/$part" &
    pids+=("$!")
    if [ "${#pids[@]}" -eq 4 ]; then
      status=0
      for pid in "${pids[@]}"; do wait "$pid" || status=1; done
      [ "$status" -eq 0 ] || { echo "Runtime part download failed" >&2; exit 2; }
      pids=()
    fi
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || { echo "Runtime part download failed" >&2; exit 2; }

  (cd "$download_dir" && /usr/bin/shasum -a 256 -c "$parts_manifest")
  : > "$download_dir/$archive"
  for part in "${parts[@]}"; do
    /bin/cat "$download_dir/$part" >> "$download_dir/$archive"
  done
  (cd "$download_dir" && /usr/bin/shasum -a 256 -c "$archive.sha256")
  ASSET="$download_dir/$archive"
else
  ASSET=$(CDPATH= cd -- "$(dirname -- "$ASSET")" && pwd)/$(basename -- "$ASSET")
  [ -f "$ASSET" ] || { echo "Asset is missing: $ASSET" >&2; exit 2; }
  if [ -f "$ASSET.sha256" ]; then
    (cd "$(dirname -- "$ASSET")" && /usr/bin/shasum -a 256 -c "$(basename -- "$ASSET").sha256")
  else
    echo "Checksum sidecar is missing: $ASSET.sha256" >&2
    exit 2
  fi
fi

destination="$RUNTIMES/$VERSION"
[ ! -e "$destination" ] || { echo "Runtime already installed: $destination" >&2; exit 2; }
temporary=$(/usr/bin/mktemp -d "$RUNTIMES/.install.XXXXXX")
cleanup_install() { [ ! -d "$temporary" ] || /bin/rm -rf -- "$temporary"; }
trap cleanup_install EXIT
expected_root="flagos-runtime-$VERSION-darwin-arm64-m5pro"
while IFS= read -r member; do
  case "$member" in
    "$expected_root"|"$expected_root/"|"$expected_root/"*) ;;
    *) echo "Archive contains an unexpected path: $member" >&2; exit 2 ;;
  esac
done < <(/usr/bin/tar -tzf "$ASSET")
/usr/bin/tar -C "$temporary" -xzf "$ASSET"
extracted="$temporary/$expected_root"
[ -x "$extracted/bin/vllm" ] || { echo "Archive layout is invalid" >&2; exit 2; }

# Import the packaged vLLM and plugin before making this version current.
/usr/bin/perl -e 'alarm shift; exec @ARGV' 120 "$extracted/bin/vllm" --version >/dev/null

/bin/mv "$extracted" "$destination"
activate_runtime "$destination"
trap - EXIT
/bin/rm -rf -- "$temporary"

echo "Installed FlagOS Runtime $VERSION at $destination"
echo "Add to your shell: export PATH=\"$CURRENT/bin:\$PATH\""
echo "Then run the standard CLI: vllm serve /path/to/model"
