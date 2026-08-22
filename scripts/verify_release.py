#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from macho_audit import audit_tree


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0-alpha.2"
RUNTIME_NAME = f"flagos-runtime-{VERSION}-darwin-arm64-m5pro"
WHEELHOUSE_NAME = f"wheelhouse-{VERSION}"
RUNTIME_ASSET = ROOT / "artifacts" / f"{RUNTIME_NAME}.tar.gz"
WHEELHOUSE_ASSET = (
    ROOT / "artifacts" / f"flagos-wheelhouse-{VERSION}-cp311-darwin-arm64.tar.gz"
)
RUNTIME_PARTS_MANIFEST = Path(str(RUNTIME_ASSET) + ".parts")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sidecar(asset: Path) -> None:
    sidecar = Path(str(asset) + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").split()
    if len(fields) < 2 or fields[0] != sha256(asset) or fields[1] != asset.name:
        raise RuntimeError(f"invalid checksum sidecar: {sidecar}")


def verify_runtime_parts() -> list[Path]:
    lines = RUNTIME_PARTS_MANIFEST.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("Runtime parts manifest is empty")

    parts: list[Path] = []
    combined = hashlib.sha256()
    for index, line in enumerate(lines):
        fields = line.split()
        expected_name = f"{RUNTIME_ASSET.name}.part-{index:03d}"
        if len(fields) != 2 or fields[1] != expected_name:
            raise RuntimeError(f"invalid Runtime part entry: {line}")
        part = RUNTIME_ASSET.parent / expected_name
        if not part.is_file() or sha256(part) != fields[0]:
            raise RuntimeError(f"invalid Runtime part: {part}")
        with part.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                combined.update(chunk)
        parts.append(part)

    if combined.hexdigest() != sha256(RUNTIME_ASSET):
        raise RuntimeError("Runtime parts do not reconstruct the validated archive")
    return parts


def normalized_member(parts: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized:
                raise RuntimeError("archive link escapes its top-level directory")
            normalized.pop()
        else:
            normalized.append(part)
    return tuple(normalized)


def check_members(archive: tarfile.TarFile, top: str) -> None:
    members = archive.getmembers()
    if not members:
        raise RuntimeError("archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise RuntimeError(f"unsafe archive path: {member.name}")
        if path.parts[0] != top:
            raise RuntimeError(f"unexpected archive top-level path: {member.name}")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            combined = target.parts if target.is_absolute() else path.parent.parts + target.parts
            resolved = normalized_member(tuple(combined))
            if not resolved or resolved[0] != top:
                raise RuntimeError(f"archive link escapes top-level path: {member.name}")


def verify_runtime_hashes(runtime: Path) -> int:
    manifest_path = runtime / "share" / "flagos" / "runtime-files.sha256.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("files", {})
    errors: list[str] = []
    checked = 0
    for relative, metadata in entries.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe hash path: {relative}")
            continue
        path = runtime / relative_path
        if not path.exists() and not path.is_symlink():
            errors.append(f"missing Runtime file: {relative}")
            continue
        if path.is_symlink():
            actual = hashlib.sha256(os.readlink(path).encode()).hexdigest()
        else:
            actual = sha256(path)
        checked += 1
        if actual != metadata.get("sha256"):
            errors.append(f"Runtime hash mismatch: {relative}")
    if checked != payload.get("file_count"):
        errors.append(
            f"Runtime hash count mismatch: checked {checked}, expected {payload.get('file_count')}"
        )
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    return checked


def verify_wheelhouse(root: Path) -> int:
    sums = root / "SHA256SUMS"
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        expected[name.lstrip("*")] = digest
    wheels = sorted(root.glob("*.whl"))
    required_prefixes = (
        "flag_gems-",
        "triton-",
        "vllm-",
        "vllm_plugin_fl-",
    )
    if len(wheels) != len(required_prefixes):
        raise RuntimeError(
            f"expected {len(required_prefixes)} wheels, found {len(wheels)}"
        )
    for prefix in required_prefixes:
        if not any(wheel.name.startswith(prefix) for wheel in wheels):
            raise RuntimeError(f"wheelhouse is missing {prefix}*")
    for wheel in wheels:
        if expected.get(wheel.name) != sha256(wheel):
            raise RuntimeError(f"wheel checksum mismatch: {wheel.name}")
    return len(wheels)


def verify_text_relocation(runtime: Path) -> int:
    """Reject text configuration and shebangs tied to the build host."""
    forbidden = (
        str(Path.home()).encode(),
        b"/private/var/folders/",
    )
    violations: list[str] = []
    checked = 0
    for path in runtime.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 8 << 20:
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        checked += 1
        relative = str(path.relative_to(runtime))
        for prefix in forbidden:
            if prefix and prefix in data:
                violations.append(f"{relative}: contains {prefix.decode()}")
        if relative.startswith("python/bin/") and data.startswith(b"#!"):
            violations.append(f"{relative}: embedded console-script shebang")
    if violations:
        raise RuntimeError("host text references: " + "; ".join(violations[:10]))
    return checked


def verify_no_release_residue(runtime: Path) -> None:
    """Reject test, debug and link-time-only files from the end-user Runtime."""
    forbidden_files = {
        Path("python/lib/python3.11/site-packages/xgrammar/lib/libxgrammar.a"),
    }
    violations: list[str] = []
    for path in runtime.rglob("*"):
        relative = path.relative_to(runtime)
        forbidden_directory = path.name in {"test", "tests"} or path.name.endswith(
            ".dSYM"
        )
        if path.is_dir() and forbidden_directory:
            violations.append(str(relative))
        elif path.is_file() and relative in forbidden_files:
            violations.append(str(relative))
    if violations:
        raise RuntimeError(
            "Runtime contains test/debug/development residue: "
            + "; ".join(violations[:10])
        )


def verify_jit_helpers(runtime: Path) -> None:
    """Ensure libtriton_jit has its relocatable Python runtime resources."""
    script_dir = runtime / "share" / "triton_jit" / "scripts"
    helpers = (script_dir / "gen_ssig.py", script_dir / "standalone_compile.py")
    missing = [str(path.relative_to(runtime)) for path in helpers if not path.is_file()]
    if missing:
        raise RuntimeError(f"Runtime is missing libtriton_jit helpers: {missing}")
    probe = subprocess.run(
        [
            str(runtime / "python" / "bin" / "python"),
            "-c",
            (
                "import pathlib,sys; "
                "sys.path.insert(0, str(pathlib.Path(sys.argv[1]))); "
                "import gen_ssig, standalone_compile"
            ),
            str(script_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if probe.returncode:
        raise RuntimeError(f"libtriton_jit helper import failed: {probe.stderr}")


def verify_m5_profile(path: Path) -> None:
    """Pin the validated multi-model routes in the shipped M5 Pro profile."""
    profile = path.read_text(encoding="utf-8")
    required = (
        "export FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL=1",
        "export FLAGGEMS_ARM_Q4_G128_PREFILL_BLOCK_M=16",
        "export FLAGGEMS_ARM_Q4_G128_PREFILL_SUBGROUP_UNROLL=1",
        "export FLAGGEMS_W8_STEALING_PREFILL=1",
        "export FLAGGEMS_W8_PREFILL_THREADS",
        "export FLAGGEMS_W8_STEALING_DECODE=1",
        "export FLAGGEMS_W8_STEALING_MIN_WORK",
        "export FLAGGEMS_W8_BODY_STEAL_CHUNK",
        "export FLAGGEMS_VLLM_FAST_APPLY=1",
    )
    missing = [setting for setting in required if setting not in profile]
    if missing:
        raise RuntimeError(f"M5 Pro profile is missing validated settings: {missing}")
    forbidden = (
        "FLAGGEMS_Q4_FUSED_LLAMA_SWIGLU_DOWN",
        "TORCH_CACHING_PRECOMPILE",
        "TORCH_STRICT_PRECOMPILE",
    )
    stale = [setting for setting in forbidden if setting in profile]
    if stale:
        raise RuntimeError(f"M5 Pro profile contains stale settings: {stale}")


def verify_model_registry() -> None:
    manifest = json.loads(
        (ROOT / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    source_lock = json.loads(
        (ROOT / "sources.lock.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 2:
        raise RuntimeError("Runtime manifest must use the multi-model schema")
    if manifest.get("version") != VERSION:
        raise RuntimeError("Runtime manifest version differs from release scripts")
    if source_lock.get("runtime_version") != VERSION:
        raise RuntimeError("source lock version differs from release scripts")
    models = manifest.get("supported_models")
    if not isinstance(models, list) or not models:
        raise RuntimeError("Runtime manifest has no supported models")
    names = [model.get("name") for model in models]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise RuntimeError("supported model names must be non-empty and unique")
    required = {
        "name",
        "architecture",
        "inference_mode",
        "modelscope_repo",
        "variants",
    }
    for model in models:
        missing = sorted(required - model.keys())
        if missing:
            raise RuntimeError(
                f"supported model {model.get('name')} is missing fields: {missing}"
            )
        variants = model["variants"]
        if not isinstance(variants, list) or not variants:
            raise RuntimeError(f"supported model {model['name']} has no variants")
        variant_names = [variant.get("name") for variant in variants]
        if any(not name for name in variant_names) or len(variant_names) != len(
            set(variant_names)
        ):
            raise RuntimeError(
                f"supported model {model['name']} has invalid variant names"
            )
        for variant in variants:
            missing_variant = {
                "name",
                "quantization",
                "acceptance_evidence",
            } - variant.keys()
            if missing_variant:
                raise RuntimeError(
                    f"model variant {model['name']}/{variant.get('name')} is "
                    f"missing fields: {sorted(missing_variant)}"
                )
            evidence = ROOT / variant["acceptance_evidence"]
            if not evidence.is_file():
                raise RuntimeError(
                    f"model variant {model['name']}/{variant['name']} has no "
                    f"acceptance evidence: {evidence}"
                )


def verify_provenance(runtime: Path) -> None:
    runtime_manifest = json.loads(
        (ROOT / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    embedded_manifest = json.loads(
        (runtime / "share" / "flagos" / "runtime-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if embedded_manifest != runtime_manifest:
        raise RuntimeError("embedded Runtime manifest differs from repository manifest")
    provenance = runtime_manifest["source_provenance"]
    if provenance["lock_sha256"] != sha256(ROOT / "sources.lock.json"):
        raise RuntimeError("source lock SHA does not match Runtime provenance")


def main() -> int:
    verify_model_registry()
    verify_m5_profile(ROOT / "profiles" / "m5-pro.env")
    check_sidecar(RUNTIME_ASSET)
    check_sidecar(WHEELHOUSE_ASSET)
    installer_sidecar = (ROOT / "install.sh.sha256").read_text(encoding="utf-8").split()
    if installer_sidecar[:2] != [sha256(ROOT / "install.sh"), "install.sh"]:
        raise RuntimeError("install.sh.sha256 is stale")

    with tempfile.TemporaryDirectory(prefix="flagos-release-verify.") as temporary:
        temp = Path(temporary)
        with tarfile.open(RUNTIME_ASSET, "r:gz") as archive:
            check_members(archive, RUNTIME_NAME)
            archive.extractall(temp)
        runtime = temp / RUNTIME_NAME
        checked_files = verify_runtime_hashes(runtime)
        verify_no_release_residue(runtime)
        checked_text_files = verify_text_relocation(runtime)
        verify_jit_helpers(runtime)
        verify_m5_profile(runtime / "share" / "flagos" / "m5-pro.env")
        verify_provenance(runtime)
        binary = audit_tree(runtime)
        if binary.get("forbidden_references"):
            raise RuntimeError("extracted Runtime contains host absolute references")
        if not binary.get("single_openmp_runtime"):
            raise RuntimeError("extracted Runtime does not have one OpenMP runtime")

        version = subprocess.run(
            [str(runtime / "bin" / "vllm"), "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if version.returncode or "0.20.2" not in version.stdout + version.stderr:
            raise RuntimeError(
                "extracted Runtime vLLM import/version probe failed: "
                + version.stderr
            )

        site = runtime / "python/lib/python3.11/site-packages"
        probe_env = dict(os.environ)
        probe_env.update(
            {
                "PYTHONHOME": str(runtime / "python"),
                "PYTHONPATH": str(site),
                "DYLD_LIBRARY_PATH": os.pathsep.join(
                    [
                        str(runtime / "lib"),
                        str(runtime / "python/lib"),
                        str(runtime / "python/readline/lib"),
                        str(runtime / "python/openssl/lib"),
                        str(site / "torch/lib"),
                    ]
                ),
            }
        )
        native_probe = subprocess.run(
            [
                str(runtime / "python/bin/python3.11"),
                "-c",
                (
                    "import torch, tvm_ffi, xgrammar; "
                    "torch.ops.load_library(r'"
                    + str(site / "flag_gems/csrc/arm/libflag_gems_arm_ops.dylib")
                    + "'); "
                    "required=('q4_linear_g128','q4_linear_g128_pair',"
                    "'w8_linear_kai','gdn_packed_decode','gdn_prefill',"
                    "'gdn_conv1d_prefill','gdn_rmsnorm_gated',"
                    "'launch_profile_start','launch_profile_stop'); "
                    "missing=[n for n in required if not hasattr(torch.ops.triton_jit_cpu,n)]; "
                    "assert not missing, missing"
                ),
            ],
            env=probe_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if native_probe.returncode:
            raise RuntimeError("native operator probe failed: " + native_probe.stderr)

        with tarfile.open(WHEELHOUSE_ASSET, "r:gz") as archive:
            check_members(archive, WHEELHOUSE_NAME)
            archive.extractall(temp)
        wheel_count = verify_wheelhouse(temp / WHEELHOUSE_NAME)

    runtime_parts = verify_runtime_parts()
    release_sums = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        release_sums[relative] = digest
    release_files = {
        RUNTIME_PARTS_MANIFEST.name: RUNTIME_PARTS_MANIFEST,
        **{part.name: part for part in runtime_parts},
        WHEELHOUSE_ASSET.name: WHEELHOUSE_ASSET,
        "install.sh": ROOT / "install.sh",
        "runtime-manifest.json": ROOT / "runtime-manifest.json",
        "sources.lock.json": ROOT / "sources.lock.json",
    }
    if set(release_sums) != set(release_files):
        raise RuntimeError("root SHA256SUMS has an unexpected Release file set")
    for relative, expected in release_sums.items():
        if sha256(release_files[relative]) != expected:
            raise RuntimeError(f"root SHA256SUMS mismatch: {relative}")

    print(
        json.dumps(
            {
                "ok": True,
                "runtime_sha256": sha256(RUNTIME_ASSET),
                "runtime_hashed_files": checked_files,
                "runtime_text_files_checked": checked_text_files,
                "macho_count": binary.get("macho_count"),
                "single_openmp_runtime": binary.get("single_openmp_runtime"),
                "wheelhouse_sha256": sha256(WHEELHOUSE_ASSET),
                "wheel_count": wheel_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
