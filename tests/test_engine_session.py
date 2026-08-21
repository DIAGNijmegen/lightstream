from types import SimpleNamespace

import pytest
import torch

from lightstream.core.engine.executors import BackwardCall, BackwardExecutor, ForwardCall, ForwardExecutor
from lightstream.core.engine.session import StreamSession


def test_stream_session_rejects_nonmatching_backward_image():
    session = StreamSession.for_forward(torch.zeros(1, 3, 8, 8), None)

    with pytest.raises(ValueError, match="does not match the pending forward session"):
        session.validate_backward_image(torch.zeros(1, 3, 9, 8))


def test_forward_executor_rejects_a_second_forward_while_session_pending():
    plan = object()
    runtime = SimpleNamespace(plan=plan)
    executor = ForwardExecutor(runtime)
    executor.pending_session = StreamSession.for_forward(torch.zeros(1, 3, 2, 2), None)

    with pytest.raises(RuntimeError, match="already pending backward"):
        executor.execute(plan, ForwardCall(torch.zeros(1, 3, 2, 2)))


def test_backward_executor_distinguishes_missing_and_consumed_sessions():
    plan = object()
    runtime = SimpleNamespace(plan=plan)
    runtime._forward_executor = ForwardExecutor(runtime)
    executor = BackwardExecutor(runtime, lambda module: False)
    call = BackwardCall(torch.zeros(1, 3, 2, 2), torch.zeros(1))

    with pytest.raises(RuntimeError, match="No pending streaming forward session"):
        executor.execute(plan, call)

    consumed = StreamSession.for_forward(call.image, None)
    consumed.consumed = True
    runtime._forward_executor.last_session = consumed
    with pytest.raises(RuntimeError, match="already been consumed"):
        executor.execute(plan, call)
