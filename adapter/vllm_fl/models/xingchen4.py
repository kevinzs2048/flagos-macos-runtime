"""XingChen4 registration over the shared, version-aware mHC/MLA runtime."""

from vllm_fl.models.telechat4 import TeleChat4ForCausalLM


class XingChen4ForCausalLM(TeleChat4ForCausalLM):
    """XingChen4 uses its hc_fn/hc_base/hc_scale checkpoint contract."""

    pass


class Xing4_0ForCausalLM(XingChen4ForCausalLM):
    """Renamed XingChen4 checkpoint with the identical mHC/MLA contract."""

    pass
