"""The runtime consumes FlagGems; operators must not import model adapters."""

import ast
from pathlib import Path

import flag_gems
from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
    DynamicW8SMELinear,
)

import qwen_image_cpu
from qwen_image_cpu.pipeline import DynamicW8SMELinear as PipelineLinear


def test_pipeline_uses_flaggems_linear_identity():
    assert PipelineLinear is DynamicW8SMELinear
    assert DynamicW8SMELinear.__module__.startswith("flag_gems.")
    assert "examples/qwen_image_cpu" not in str(Path(qwen_image_cpu.__file__))


def test_generic_operators_do_not_depend_on_model_package():
    root = Path(flag_gems.__file__).parent / "runtime/backend/_arm"
    paths = list((root / "quantized_linear/sme2").glob("*.py"))
    paths.append(root / "ops/image_cpu_candidates.py")
    assert len(paths) >= 12
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("qwen_image_cpu", "diffusers")
                ), path
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith(("qwen_image_cpu", "diffusers"))
                    for alias in node.names
                ), path
