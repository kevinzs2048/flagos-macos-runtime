#!/usr/bin/env python3
"""Materialize the exact source trees pinned by sources.lock.json.

Local clean checkouts can be supplied with FLAGOS_*_SOURCE variables.  Any
remaining component is fetched from its locked URL and commit.  In both cases
the commit and Git tree must match the lock before files reach build/sources.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "sources.lock.json"
OUTPUT_ROOT = ROOT / "build" / "sources"
CACHE_ROOT = ROOT / "build" / "source-cache"
COMMAND_TIMEOUT = 300

COMPONENTS = {
    "vllm": ("vllm", "FLAGOS_VLLM_SOURCE"),
    "triton_cpu": ("triton-cpu", "FLAGOS_TRITON_SOURCE"),
    "flag_gems": ("FlagGems", "FLAGOS_FLAGGEMS_SOURCE"),
    "libtriton_jit": ("libtriton_jit", "FLAGOS_LIBTRITON_JIT_SOURCE"),
    "vllm_plugin_fl": ("vllm-plugin-FL", "FLAGOS_PLUGIN_SOURCE"),
}


def run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=COMMAND_TIMEOUT,
    )
    return result.stdout.strip() if capture else ""


def git(repo: Path, *args: str) -> str:
    return run("git", "-C", str(repo), *args, capture=True)


def verify_repo(repo: Path, name: str, commit: str, tree: str, clean: bool) -> None:
    if not (repo / ".git").exists() and not (repo / "HEAD").is_file():
        raise RuntimeError(f"{name}: not a Git checkout: {repo}")
    actual_commit = git(repo, "rev-parse", "HEAD")
    actual_tree = git(repo, "rev-parse", "HEAD^{tree}")
    if actual_commit != commit:
        raise RuntimeError(
            f"{name}: HEAD is {actual_commit}, sources.lock.json requires {commit}"
        )
    if actual_tree != tree:
        raise RuntimeError(
            f"{name}: tree is {actual_tree}, sources.lock.json requires {tree}"
        )
    if clean:
        status = git(repo, "status", "--porcelain", "--untracked-files=normal")
        if status:
            raise RuntimeError(f"{name}: local source tree is dirty: {repo}")


def fetch_repo(name: str, metadata: dict[str, str]) -> Path:
    repo = CACHE_ROOT / f"{name}.git"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run("git", "init", "--bare", str(repo))
    remotes = git(repo, "remote").splitlines()
    if "origin" in remotes:
        run("git", "-C", str(repo), "remote", "set-url", "origin", metadata["url"])
    else:
        run("git", "-C", str(repo), "remote", "add", "origin", metadata["url"])
    run(
        "git",
        "-C",
        str(repo),
        "fetch",
        "--force",
        "--depth=1",
        "origin",
        metadata["commit"],
    )
    run("git", "-C", str(repo), "update-ref", "HEAD", metadata["commit"])
    verify_repo(repo, name, metadata["commit"], metadata["tree"], clean=False)
    return repo


def archive(repo: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True)
    producer = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        stdout=subprocess.PIPE,
    )
    assert producer.stdout is not None
    try:
        consumer = subprocess.run(
            ["/usr/bin/tar", "-xf", "-", "-C", str(destination)],
            stdin=producer.stdout,
            check=False,
            timeout=COMMAND_TIMEOUT,
        )
    finally:
        producer.stdout.close()
    try:
        producer_result = producer.wait(timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        producer.kill()
        producer.wait()
        raise RuntimeError(f"git archive timed out: {repo}") from None
    if producer_result or consumer.returncode:
        raise RuntimeError(f"failed to materialize {repo}")


def main() -> int:
    source_lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked = source_lock.get("components", {})
    if set(locked) != set(COMPONENTS):
        raise RuntimeError("sources.lock.json has an unexpected component set")

    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="sources.materializing.", dir=OUTPUT_ROOT.parent)
    )
    try:
        for name, (directory, variable) in COMPONENTS.items():
            metadata = locked[name]
            override = os.environ.get(variable)
            if override:
                repo = Path(override).expanduser().resolve()
                verify_repo(
                    repo, name, metadata["commit"], metadata["tree"], clean=True
                )
                origin = f"local:{repo}"
            else:
                try:
                    repo = fetch_repo(name, metadata)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"{name}: locked commit {metadata['commit']} could not "
                        f"be fetched from {metadata['url']}; publish that commit "
                        f"or set {variable} to a clean matching checkout"
                    ) from error
                origin = metadata["url"]
            archive(repo, metadata["commit"], temporary / directory)
            print(f"materialized {name} {metadata['commit'][:12]} from {origin}")

        stamp = {
            "schema_version": 1,
            "lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
            "components": {
                name: {
                    "commit": locked[name]["commit"],
                    "tree": locked[name]["tree"],
                }
                for name in COMPONENTS
            },
        }
        (temporary / ".materialized.json").write_text(
            json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)
        temporary.replace(OUTPUT_ROOT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"sources ready: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"source materialization failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
