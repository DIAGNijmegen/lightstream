"""Per-call execution boundaries for the transitional StreamingCNN facade."""

from dataclasses import dataclass
from typing import Any

import torch

from .configuration import StreamingPlan


@dataclass(frozen=True)
class ForwardCall:
    image: torch.Tensor
    result_on_cpu: bool = False
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class BackwardCall:
    image: torch.Tensor
    gradient: Any
    mask: torch.Tensor | None = None


class ForwardExecutor:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, plan: StreamingPlan, context: ForwardCall):
        if plan is not self.runtime.plan:
            raise ValueError("ForwardExecutor received a plan for a different runtime")
        return self.runtime._forward_impl(context.image, context.result_on_cpu, context.mask)


class BackwardExecutor:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, plan: StreamingPlan, context: BackwardCall):
        if plan is not self.runtime.plan:
            raise ValueError("BackwardExecutor received a plan for a different runtime")
        return self.runtime._backward_impl(context.image, context.gradient, context.mask)
