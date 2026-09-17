# Copyright (c) 2025 BAAI. All rights reserved.

import logging
import os
import platform
import sys
from importlib import metadata

# torch.float4_e2m1fn_x2 exists only in CUDA builds of PyTorch 2.7+.
# vllm.ir.tolerances references it at module level, so we inject a sentinel
# before any vllm.ir import can happen.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
else:
    import torch as _torch
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
del _torch

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def _is_arm_cpu_build() -> bool:
    """Return whether this is an AArch64 vLLM CPU build.

    Architecture alone is insufficient: accelerator-enabled vLLM wheels also
    run on AArch64 hosts and must retain the original FL platform.
    """
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return False
    try:
        return "cpu" in metadata.version("vllm").lower()
    except metadata.PackageNotFoundError:
        return os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "cpu"


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def _patch_flash_attn_import():
    """Stub vllm.vllm_flash_attn if CUDA flash attention C extensions are missing."""
    import sys
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    try:
        import vllm.vllm_flash_attn  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("vllm.vllm_flash_attn")
        stub.FA2_AVAILABLE = False
        stub.FA3_AVAILABLE = False
        stub.fa_version_unsupported_reason = lambda *a, **kw: "flash_attn C extensions not available"
        stub.flash_attn_varlen_func = None
        stub.get_scheduler_metadata = None
        stub.is_fa_version_supported = lambda *a, **kw: False
        sys.modules["vllm.vllm_flash_attn"] = stub


def _patch_custom_ops():
    """Register torch.ops._C op schemas when vllm._C is unavailable."""
    try:
        import vllm._C  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
    register_op_schemas()


def register():
    """Register the FL platform."""
    # PlatformFL is accelerator-shaped. ARM CPU uses vLLM's native CPU
    # platform plus the FlagGems runtime installed by register_model().
    if _is_arm_cpu_build():
        logger.info("[vllm_fl] ARM CPU -> FL CPU platform (native-backed)")
        return "vllm_fl.platform_cpu.CpuPlatformFL"

    _patch_custom_ops()
    _patch_flash_attn_import()
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"

def register_quant_linear():
    from vllm.platforms import current_platform
    # vllm.model_executor.kernels.linear triggers cutlass_scaled_mm_supports_fp8
    # at module level, which requires torch.ops._C — not available on MUSA.
    if current_platform.device_type == "musa":
        return
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm.platforms import current_platform
    # fused_moe import chain triggers cutlass_scaled_mm_supports_fp8 on MUSA
    if current_platform.device_type == "musa":
        return
    from vllm_fl.utils import is_oot_enabled
    if not is_oot_enabled():
        return
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()

def register_model():
    """Register FL-specific models not yet upstream."""
    from vllm.platforms import current_platform
    if current_platform.device_type == "cpu" and _is_arm_cpu_build():
        from vllm.model_executor.models import ModelRegistry

        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.telechat4 import TeleChat4Config
        from vllm_fl.configs.xingchen4 import XingChen4Config, Xing4_0Config
        from vllm_fl.runtime_families import is_mhc_mla_w4a8_family

        _CONFIG_REGISTRY["telechat4"] = TeleChat4Config
        _CONFIG_REGISTRY["xingchen4"] = XingChen4Config
        _CONFIG_REGISTRY["xing4_0"] = Xing4_0Config
        ModelRegistry.register_model(
            "TeleChat4ForCausalLM",
            "vllm_fl.models.telechat4:TeleChat4ForCausalLM",
        )
        ModelRegistry.register_model(
            "XingChen4ForCausalLM",
            "vllm_fl.models.xingchen4:XingChen4ForCausalLM",
        )
        ModelRegistry.register_model(
            "Xing4_0ForCausalLM",
            "vllm_fl.models.xingchen4:Xing4_0ForCausalLM",
        )
        # The TeleChat4 draft checkpoint carries one extra MTP layer at
        # ``model.layers.40``. Register its architecture before vLLM builds
        # the draft ModelConfig, keeping the target W4A8 model untouched.
        ModelRegistry.register_model(
            "TeleChat4MTPModel",
            "vllm_fl.spec_decode.telechat4_mtp:TeleChat4MTP",
        )
        ModelRegistry.register_model(
            "XingChen4MTPModel",
            "vllm_fl.spec_decode.telechat4_mtp:XingChen4MTP",
        )
        ModelRegistry.register_model(
            "Xing4_0MTPModel",
            "vllm_fl.spec_decode.telechat4_mtp:XingChen4MTP",
        )
        from vllm_fl.spec_decode.telechat4_mtp import (
            install_telechat4_mtp_config_override,
            install_telechat4_mtp_safetensors_loader,
        )

        install_telechat4_mtp_config_override()
        install_telechat4_mtp_safetensors_loader()

        runtime_family = os.environ.get(
            "FL_CPU_RUNTIME_FAMILY", "qwen"
        ).lower()
        if (
            is_mhc_mla_w4a8_family(runtime_family)
            and os.environ.get("FL_CPU_TELECHAT4_MLA", "0").lower()
            in {"1", "true", "on"}
        ):
            # TeleChat4 has DeepSeek-V3 MLA dimensions, but its model_type is
            # not in vLLM's recognizer. Keep this process-local and dynamically
            # family-gated so W4A16 and Qwen configurations remain unchanged.
            from vllm.transformers_utils.model_arch_config_convertor import (
                ModelArchConfigConvertorBase,
            )

            method = ModelArchConfigConvertorBase.is_deepseek_mla
            if not getattr(method, "_telechat4_w4a8_patch", False):
                original = method

                def _is_deepseek_mla(self):
                    enabled = (
                        is_mhc_mla_w4a8_family()
                        and os.environ.get("FL_CPU_TELECHAT4_MLA", "0").lower()
                        in {"1", "true", "on"}
                    )
                    if (
                        enabled
                        and getattr(self.hf_text_config, "model_type", None)
                        in {
                            "telechat4",
                            "telechat4_mtp",
                            "xingchen4",
                            "xingchen4_mtp",
                            "xing4_0",
                            "xing4_0_mtp",
                        }
                    ):
                        return (
                            getattr(self.hf_text_config, "kv_lora_rank", None)
                            is not None
                        )
                    return original(self)

                _is_deepseek_mla._telechat4_w4a8_patch = True
                ModelArchConfigConvertorBase.is_deepseek_mla = _is_deepseek_mla

        if runtime_family == "telechat4":
            # WNA16 is the W4A16 compatibility path.  Do not install its MoE
            # selector for TeleChat4 W4A8 or the established Qwen W4A8 route.
            from vllm_fl.patches.cpu_wna16_moe import (
                install_cpu_wna16_moe_selection,
            )

            install_cpu_wna16_moe_selection()

        ModelRegistry.register_model(
            "DFlash2DraftModel",
            "vllm_fl.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
        )
        # vLLM inspects lazy model classes in a short-lived registry
        # subprocess. That process only needs registrations; importing the
        # full FlagGems runtime there allocates hundreds of process locks and
        # can exhaust Darwin's named semaphore pool before the real engine
        # starts.
        if os.path.basename(sys.argv[0]) == "registry.py":
            return
        # INT8 modes have priority when explicitly enabled. The production
        # backend is FlagGems/libtriton_jit; torchpack is retained as an
        # explicit torch-native diagnostic fallback.
        int8_enabled = os.environ.get("FL_CPU_INT8", "0").lower()
        if int8_enabled not in {"0", "1", "false", "true"}:
            raise ValueError("FL_CPU_INT8 must be one of: 0, 1, false, true")
        if int8_enabled in {"1", "true"}:
            int8_backend = os.environ.get(
                "FL_CPU_INT8_BACKEND", "libtriton_jit"
            ).lower()
            if int8_backend == "torchpack":
                from vllm_fl.ops.cpu_int8_pack import enable_int8
            elif int8_backend == "libtriton_jit":
                # FlagGems owns online W8 packing and the native Qwen GDN
                # operations behind the same process-global operator library.
                from vllm_fl.ops.cpu_qwen_runtime import enable_qwen_runtime

                enable_qwen_runtime()
                return
            else:
                raise ValueError(
                    "FL_CPU_INT8_BACKEND must be 'libtriton_jit' "
                    "or 'torchpack'"
                )

            enable_int8()
            return
        enabled = os.environ.get("FL_CPU_INT4", "0").lower()
        if enabled not in {"0", "1", "false", "true"}:
            raise ValueError("FL_CPU_INT4 must be one of: 0, 1, false, true")
        if enabled in {"1", "true"}:
            configured = os.environ.get(
                "FL_CPU_INT4_BACKEND", "libtriton_jit"
            ).lower()
            if configured != "libtriton_jit":
                raise ValueError(
                    "FL_CPU_INT4_BACKEND must be 'libtriton_jit'"
                )
            if runtime_family == "telechat4":
                from vllm_fl.ops.cpu_telechat4_runtime import (
                    enable_telechat4_runtime,
                )

                enable_telechat4_runtime()
            elif is_mhc_mla_w4a8_family(runtime_family):
                from vllm_fl.ops.cpu_telechat4_w4a8_runtime import (
                    enable_telechat4_w4a8_runtime,
                )

                enable_telechat4_w4a8_runtime()
            elif runtime_family == "qwen":
                from vllm_fl.ops.cpu_qwen_runtime import enable_qwen_runtime

                enable_qwen_runtime()
            else:
                raise ValueError(
                    "FL_CPU_RUNTIME_FAMILY must be 'qwen', 'telechat4', "
                    "'telechat4_w4a8', or 'xingchen4_w4a8'"
                )
        else:
            logger.info("[vllm_fl] FL_CPU_INT4=0 -> bf16 (int4 op skipped)")
        return

    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
