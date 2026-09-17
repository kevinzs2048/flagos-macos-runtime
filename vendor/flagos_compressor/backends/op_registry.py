from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


OpFunc = Callable[..., Any]


@dataclass(frozen=True)
class OpImpl:
    op_name: str
    backend_name: str
    func: OpFunc


class OpRegistry:
    def __init__(self) -> None:
        self._impls: dict[tuple[str, str], OpImpl] = {}

    def register(self, op_name: str, backend_name: str, func: OpFunc) -> None:
        self._impls[(op_name, backend_name)] = OpImpl(op_name, backend_name, func)

    def resolve(self, op_name: str, backend_name: str) -> OpImpl | None:
        return self._impls.get((op_name, backend_name))

    def has(self, op_name: str, backend_name: str) -> bool:
        return (op_name, backend_name) in self._impls

    def available_backends(self, op_name: str) -> list[str]:
        return sorted(backend for op, backend in self._impls if op == op_name)


GLOBAL_OP_REGISTRY = OpRegistry()


def register_op(op_name: str, backend_name: str) -> Callable[[OpFunc], OpFunc]:
    def decorator(func: OpFunc) -> OpFunc:
        GLOBAL_OP_REGISTRY.register(op_name, backend_name, func)
        return func

    return decorator

