"""XingChen4 configuration compatibility for the ARM CPU runtime."""

from vllm_fl.configs.telechat4 import TeleChat4Config


class XingChen4Config(TeleChat4Config):
    """Normalize XingChen4's released mHC aliases to vLLM field names."""

    model_type = "xingchen4"

    def __init__(
        self,
        hc_mult: int | None = None,
        hc_sinkhorn_iters: int | None = None,
        hc_eps: float | None = None,
        num_residual_streams: int | None = None,
        mhc_sinkhorn_iterations: int | None = None,
        mhc_norm_eps: float | None = None,
        mhc_pre_eps: float | None = None,
        **kwargs,
    ) -> None:
        streams = (
            num_residual_streams
            if num_residual_streams is not None
            else hc_mult if hc_mult is not None else 4
        )
        sinkhorn_iterations = (
            mhc_sinkhorn_iterations
            if mhc_sinkhorn_iterations is not None
            else hc_sinkhorn_iters if hc_sinkhorn_iters is not None else 20
        )
        norm_eps = (
            mhc_norm_eps
            if mhc_norm_eps is not None
            else hc_eps if hc_eps is not None else 1e-6
        )
        pre_eps = (
            mhc_pre_eps
            if mhc_pre_eps is not None
            else hc_eps if hc_eps is not None else norm_eps
        )
        super().__init__(
            num_residual_streams=int(streams),
            mhc_sinkhorn_iterations=int(sinkhorn_iterations),
            **kwargs,
        )
        self.mhc_norm_eps = float(norm_eps)
        self.mhc_pre_eps = float(pre_eps)
        self.hc_mult = self.num_residual_streams
        self.hc_sinkhorn_iters = self.mhc_sinkhorn_iterations
        self.hc_eps = self.mhc_norm_eps


class Xing4_0Config(XingChen4Config):
    """September 16 naming alias; preserve the original model identity."""

    model_type = "xing4_0"

    def __init__(self, **kwargs) -> None:
        # Transformers 5 generates a dataclass initializer for subclasses
        # without their own __init__, bypassing the inherited mHC aliases.
        super().__init__(**kwargs)
