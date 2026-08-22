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

NATIVE_OPERATOR_SMOKE = r"""
import os

import torch
import tvm_ffi  # noqa: F401
import xgrammar  # noqa: F401

from flag_gems.csrc.arm import configure_runtime
from flag_gems.runtime.backend._arm.q4.linear import (
    pack_rhs_qsi4c128p_asym,
    pack_rhs_w8_symmetric,
)

torch.ops.load_library(str(configure_runtime().resolve()))
required = (
    "q4_linear_g128",
    "q4_linear_g128_pair",
    "w8_linear_kai",
    "gdn_packed_decode",
    "gdn_prefill",
    "gdn_conv1d_prefill",
    "gdn_rmsnorm_gated",
    "launch_profile_start",
    "launch_profile_stop",
)
missing = [
    name for name in required if not hasattr(torch.ops.triton_jit_cpu, name)
]
assert not missing, missing

torch.manual_seed(20260822)
torch.set_num_threads(2)
n, k = 64, 128

q4_weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
q4_scale = 0.001 + 0.02 * torch.rand(n, k // 128)
q4_rhs = pack_rhs_qsi4c128p_asym(q4_weight, q4_scale)
q4_x = torch.randn((32, k), dtype=torch.bfloat16)
os.environ["FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL"] = "0"
q4_regular = torch.ops.triton_jit_cpu.q4_linear_g128(q4_x, q4_rhs, n, k)
os.environ["FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL"] = "1"
os.environ["FLAGGEMS_ARM_Q4_G128_STEAL_CHUNK"] = "2"
q4_stealing = torch.ops.triton_jit_cpu.q4_linear_g128(q4_x, q4_rhs, n, k)
torch.testing.assert_close(q4_stealing, q4_regular, rtol=0, atol=0)

w8_weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
w8_scale = 0.001 + 0.02 * torch.rand(n)
w8_rhs = pack_rhs_w8_symmetric(w8_weight, w8_scale)
w8_x = torch.randn((32, k), dtype=torch.bfloat16)
os.environ["FLAGGEMS_W8_STEALING_PREFILL"] = "0"
w8_regular = torch.ops.triton_jit_cpu.w8_linear_kai(w8_x, w8_rhs, n, k)
os.environ["FLAGGEMS_W8_STEALING_PREFILL"] = "1"
os.environ["FLAGGEMS_W8_PREFILL_STEAL_CHUNK"] = "2"
w8_stealing = torch.ops.triton_jit_cpu.w8_linear_kai(w8_x, w8_rhs, n, k)
torch.testing.assert_close(w8_stealing, w8_regular, rtol=0, atol=0)

values = w8_x.to(torch.float32)
absmax = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
quantized = torch.round(values * (127.0 / absmax)).clamp_(-127, 127)
reference = (
    (quantized @ w8_weight.to(torch.float32).T)
    * ((absmax / 127.0) * w8_scale.to(torch.float32)[None, :])
).to(torch.bfloat16)
torch.testing.assert_close(w8_regular, reference, rtol=0.02, atol=0.125)
"""


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
    expected = {
        "FLAGGEMS_ARM_Q4_G128_PREFILL_BLOCK_M": "16",
        "FLAGGEMS_ARM_Q4_G128_PREFILL_SUBGROUP_UNROLL": "1",
        "FLAGGEMS_ARM_Q4_G128_STEALING_DECODE": "1",
        "FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL": "1",
        "FLAGGEMS_ARM_Q4_G128_STEAL_CHUNK": "2",
        "FLAGGEMS_ARM_Q4_G128_SWIGLU_STEALING": "0",
        "FLAGGEMS_ARM_Q4_STRICT": "1",
        "FLAGGEMS_GDN_PREFILL_THREADS": "16",
        "FLAGGEMS_GDN_TRITON_BLOCK_KEY": "32",
        "FLAGGEMS_GDN_TRITON_DECODE": "1",
        "FLAGGEMS_GDN_TRITON_THREADS": "14",
        "FLAGGEMS_Q4_DECODE_PARTITIONS": "auto",
        "FLAGGEMS_Q4_PREFILL_THREADS": "18",
        "FLAGGEMS_Q4_STEAL_CHUNK": "32",
        "FLAGGEMS_VENDOR": "arm",
        "FLAGGEMS_VLLM_FAST_APPLY": "1",
        "FLAGGEMS_W8_BODY_STEAL_CHUNK": "32",
        "FLAGGEMS_W8_PREFILL_STEAL_CHUNK": "2",
        "FLAGGEMS_W8_PREFILL_THREADS": "16",
        "FLAGGEMS_W8_STEALING_DECODE": "1",
        "FLAGGEMS_W8_STEALING_MIN_WORK": "0",
        "FLAGGEMS_W8_STEALING_PREFILL": "1",
        "FLAGGEMS_W8_STEAL_CHUNK": "64",
        "FLAGOS_INDUCTOR_TOKEN_PARALLEL_GUARD": "1",
        "FL_CPU_INT4": "1",
        "FL_CPU_INT4_BACKEND": "libtriton_jit",
        "FL_CPU_UNIPROC": "1",
        "FL_INT8_LMHEAD": "1",
        "KMP_BLOCKTIME": "20",
        "OMP_NUM_THREADS": "14",
        "TOKENIZERS_PARALLELISM": "false",
        "TRITON_JIT_CPU_DYNAMIC_CHUNK": "32",
        "TRITON_JIT_CPU_DYNAMIC_KERNEL": "_q4_prefill_asym_g128_i8mm_kernel",
        "TRITON_LOCAL_LIBOMP_PATH": "/runtime",
        "VLLM_CPU_ATTN_SPLIT_KV": "0",
        "VLLM_CPU_KVCACHE_SPACE": "2",
        "VLLM_CPU_OMP_THREADS_BIND": "nobind",
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_PLUGINS": "fl",
    }
    probe = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; /usr/bin/env',
            "verify-profile",
            str(path.resolve()),
        ],
        env={"FLAGOS_RUNTIME_ROOT": "/runtime"},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if probe.returncode:
        raise RuntimeError(f"M5 Pro profile cannot be loaded: {probe.stderr}")
    exported = dict(
        line.split("=", 1) for line in probe.stdout.splitlines() if "=" in line
    )
    prefixes = ("FLAGGEMS_", "FLAGOS_", "FL_", "TRITON_", "VLLM_")
    managed = {
        key: value
        for key, value in exported.items()
        if key.startswith(prefixes)
        or key in {"KMP_BLOCKTIME", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"}
    }
    managed.pop("FLAGOS_RUNTIME_ROOT", None)
    if managed != expected:
        missing = sorted(expected.keys() - managed.keys())
        unexpected = sorted(managed.keys() - expected.keys())
        mismatched = {
            key: {"expected": expected[key], "actual": managed[key]}
            for key in sorted(expected.keys() & managed.keys())
            if expected[key] != managed[key]
        }
        raise RuntimeError(
            "M5 Pro profile differs from the validated environment: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )


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
                "FLAGGEMS_VENDOR": "arm",
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
        for variable in (
            "FLAGGEMS_LIBTRITON_JIT_Q4_OP",
            "FLAGGEMS_Q4_KERNEL_SOURCE",
            "FLAGGEMS_W8_KERNEL_SOURCE",
        ):
            probe_env.pop(variable, None)
        native_probe = subprocess.run(
            [
                str(runtime / "python/bin/python3.11"),
                "-c",
                NATIVE_OPERATOR_SMOKE,
            ],
            env=probe_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
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
