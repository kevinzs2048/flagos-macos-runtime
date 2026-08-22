#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=0.1.0-alpha.2
ASSET="$ROOT/artifacts/flagos-runtime-$VERSION-darwin-arm64-m5pro.tar.gz"
TEST_ROOT=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/flagos-install-smoke.XXXXXX")

cleanup() {
  case "$TEST_ROOT" in
    "${TMPDIR:-/tmp}"/flagos-install-smoke.*) /bin/rm -rf -- "$TEST_ROOT" ;;
    *) echo "Refusing unsafe smoke-test cleanup: $TEST_ROOT" >&2 ;;
  esac
}
trap cleanup EXIT

[ -f "$ASSET" ] || { echo "Runtime asset is missing: $ASSET" >&2; exit 2; }
[ -f "$ASSET.sha256" ] || { echo "Runtime checksum is missing" >&2; exit 2; }

export HOME="$TEST_ROOT/home"
export FLAGOS_INSTALL_ROOT="$TEST_ROOT/FlagOS"
/bin/mkdir -p "$HOME"

/bin/bash "$ROOT/install.sh" --asset "$ASSET"
COMMAND="$FLAGOS_INSTALL_ROOT/current/bin/vllm"
[ -x "$COMMAND" ] || { echo "installed vllm launcher is missing" >&2; exit 2; }

VERSION_OUTPUT=$(/usr/bin/perl -e 'alarm shift; exec @ARGV' 120 "$COMMAND" --version 2>&1)
case "$VERSION_OUTPUT" in
  *0.20.2*) ;;
  *) echo "installed Runtime reports the wrong vLLM version: $VERSION_OUTPUT" >&2; exit 2 ;;
esac
# A first launch from a freshly extracted Runtime can spend more than two
# minutes in macOS cold-file validation while importing the full vLLM command
# graph. Keep this bounded, but leave enough room for the observed 143-second
# cold path on the release host.
/usr/bin/perl -e 'alarm shift; exec @ARGV' 300 "$COMMAND" serve --help >/dev/null

/bin/bash "$ROOT/install.sh" --rollback "$VERSION" >/dev/null
if /bin/bash "$ROOT/install.sh" --uninstall "$VERSION" \
  >"$TEST_ROOT/uninstall.log" 2>&1
then
  echo "installer removed the active Runtime" >&2
  exit 2
fi
/usr/bin/grep -q "Refusing to uninstall the active Runtime" \
  "$TEST_ROOT/uninstall.log"

/bin/ln -s "$ASSET" "$TEST_ROOT/no-sidecar.tar.gz"
if /bin/bash "$ROOT/install.sh" \
  --asset "$TEST_ROOT/no-sidecar.tar.gz" --version 0.1.0-test \
  >"$TEST_ROOT/missing-sidecar.log" 2>&1
then
  echo "installer accepted an asset without a checksum sidecar" >&2
  exit 2
fi
/usr/bin/grep -q "Checksum sidecar is missing" \
  "$TEST_ROOT/missing-sidecar.log"

if FLAGOS_INSTALL_ROOT="$TEST_ROOT/Path With Spaces" \
  /bin/bash "$ROOT/install.sh" --asset "$ASSET" --version 0.1.0-space-test \
  >"$TEST_ROOT/space-path.log" 2>&1
then
  echo "installer accepted a Runtime path containing whitespace" >&2
  exit 2
fi
/usr/bin/grep -q "Runtime install path must not contain whitespace" \
  "$TEST_ROOT/space-path.log"

echo "Installed standard vLLM workflow: PASS"
