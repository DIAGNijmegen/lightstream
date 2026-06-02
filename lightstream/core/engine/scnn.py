"""Streaming CNN engine orchestration."""
import copy
import logging

import torch
import torch.backends

from lightstream.core.reducer import BaseReducer
from lightstream.core.scnn.utils import H_DIM, W_DIM
from lightstream.core.scnn.streamingconv import StreamingConv2d
from lightstream.core.scnn.streamingupsample import StreamingUpsample2d

from .backward import BackwardMixin
from .config import BackwardContext, CompiledPlan, ForwardContext, StreamingConfig, TileSpec
from .conversion import ConversionMixin
from .forward import ForwardMixin
from .planner import PlannerMixin
from .reducers import ReducerMixin
from .statistics import StatisticsMixin


logger = logging.getLogger(__name__)


class StreamingCNN(
    ForwardMixin,
    BackwardMixin,
    ReducerMixin,
    ConversionMixin,
    StatisticsMixin,
    PlannerMixin,
    torch.nn.Module,
):
    def __init__(
        self,
        stream_module,
        tile_shape,
        verbose=False,
        deterministic=False,
        saliency=False,
        eps=1e-5,
        copy_to_gpu=True,
        dtype=None,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
        mean=None,
        std=None,
        state_dict=None,
    ):
        """
        Parameters:
            stream_module (torch.nn.Module): modules containing the to be streamed layers
            tile_shape (tuple, NCHW): size of the to be streamed tiles
            verbose (bool): if True, print setup-time tile statistics and progress output
                during configuration. Per-forward tiling diagnostics are emitted through this
                module's logger at DEBUG level (default is False).
            deterministic (bool): whether to use the deterministic algorithms for cudnn
            saliency (bool): will gather the gradients of the input image (saliency map)
            eps (float): epsilon error to compare floating values
        """
        super().__init__()
        global H_DIM, W_DIM
        self.stream_module = stream_module
        self.verbose = verbose
        self.deterministic = deterministic
        self.eps = eps
        self.device = next(stream_module.parameters()).device
        self.dtype = next(stream_module.parameters()).dtype
        if dtype is not None:
            self.dtype = dtype
        self.tile_shape = tile_shape
        self.gather_input_gradient = saliency
        self.copy_to_gpu = copy_to_gpu
        self.statistics_on_cpu = statistics_on_cpu

        if mean is not None and not isinstance(mean, torch.Tensor):
            mean = torch.Tensor(mean)[:, None, None]

        if std is not None and not isinstance(std, torch.Tensor):
            std = torch.Tensor(std)[:, None, None]

        self.mean = mean if mean is not None else torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = std if std is not None else torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        self.should_normalize = normalize_on_gpu

        self.compiled_plan = CompiledPlan(stream_network=self)
        self._tile_output_shape = None
        self._tile_output_shapes = None
        self._tile_output_lost = None
        self._output_stride_per_output = None
        self._output_spec = None
        self._module_stats = {}
        self._saved_tensors = {}
        self.debug_reducer_replay = False
        self.debug_forward_sentinel_check = False
        self._hooks = []
        self._last_forward_tiles = []
        self._streaming_reducers = []
        self._reducer_head_map = {}
        self._reducer_input_indices = {}
        self._active_reducer_mask = None

        if state_dict is None:
            self._configure()
        else:
            self.load_tile_cache(state_dict)


    @property
    def _tile_output_shape(self):
        return self.compiled_plan.plan_state.tile_output_shape

    @_tile_output_shape.setter
    def _tile_output_shape(self, value):
        self.compiled_plan.plan_state.tile_output_shape = value

    @property
    def _tile_output_shapes(self):
        return self.compiled_plan.plan_state.tile_output_shapes

    @_tile_output_shapes.setter
    def _tile_output_shapes(self, value):
        self.compiled_plan.plan_state.tile_output_shapes = value

    @property
    def _tile_output_lost(self):
        return self.compiled_plan.plan_state.tile_output_lost

    @_tile_output_lost.setter
    def _tile_output_lost(self, value):
        self.compiled_plan.plan_state.tile_output_lost = value

    @property
    def _output_stride_per_output(self):
        return self.compiled_plan.plan_state.output_stride_per_output

    @_output_stride_per_output.setter
    def _output_stride_per_output(self, value):
        self.compiled_plan.plan_state.output_stride_per_output = value

    @property
    def _output_spec(self):
        return self.compiled_plan.plan_state.output_spec

    @_output_spec.setter
    def _output_spec(self, value):
        self.compiled_plan.plan_state.output_spec = value

    @property
    def _reducer_head_map(self):
        return self.compiled_plan.session_state.reducer_head_map

    @_reducer_head_map.setter
    def _reducer_head_map(self, value):
        self.compiled_plan.session_state.reducer_head_map = value

    @property
    def _reducer_input_indices(self):
        return self.compiled_plan.session_state.reducer_input_indices

    @_reducer_input_indices.setter
    def _reducer_input_indices(self, value):
        self.compiled_plan.session_state.reducer_input_indices = value

    @property
    def _last_forward_tiles(self):
        return self.compiled_plan.session_state.last_forward_tiles

    @_last_forward_tiles.setter
    def _last_forward_tiles(self, value):
        self.compiled_plan.session_state.last_forward_tiles = value

    @property
    def _active_reducer_mask(self):
        return self.compiled_plan.session_state.active_reducer_mask

    @_active_reducer_mask.setter
    def _active_reducer_mask(self, value):
        self.compiled_plan.session_state.active_reducer_mask = value

    def _print_verbose(self, *args: object, **kwargs: object) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def _configure(self):
        # Save current model and cudnn flags, since we need to change them and restore later
        state_dict = self._save_parameters()
        (old_deterministic_flag, old_benchmark_flag) = self._set_cudnn_flags_to_determistic()
        self._reset_parameters_to_constant()

        # Add hooks to each layer to gather statistics
        self._add_hooks_for_statistics()
        self._set_reducer_passthrough(True)

        # We need to temporary store statistics per layer to keep track of the
        # total output stride at each layer
        self._stats_per_grad_fn = {}

        # TODO; temp hack for tile sizes too big on gpu,
        # we need float32 precision
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cpu()
            self.device = torch.device("cpu")  # type:ignore

        # Create all-ones tile
        tile = torch.ones(self.tile_shape, dtype=self.dtype, requires_grad=True, device=self.device)

        self._gather_forward_statistics(tile)
        self._print_verbose("")
        self._gather_backward_statistics(tile)

        # TODO; temp hack for tile sizes too big on gpu,
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cuda()
            self.device = torch.device("cuda")  # type:ignore

        # Remove all hooks and add hooks for correcting gradients
        # during lightstream
        self._remove_hooks()
        self._set_reducer_passthrough(False)
        #
        self._restore_parameters(state_dict)
        self._capture_public_output_spec()
        self._streaming_reducers = []
        self.stream_module = self._convert_modules_for_streaming(self.stream_module)
        self._add_hooks_for_streaming()

        # Remove temporary data
        self._saved_tensors = {}
        del self._stats_per_grad_fn

        # Zero the gradients
        for param in self.stream_module.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        self._set_cudnn_flags(old_deterministic_flag, old_benchmark_flag)
        del state_dict

    def _capture_public_output_spec(self) -> None:
        """Capture user-facing output structure with reducers in normal (non-passthrough) mode."""
        spec_tile = torch.ones(self.tile_shape, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self.stream_module(spec_tile)
        _, output_spec = self._flatten_output_structure(output)
        self._output_spec = output_spec

    def _reset_parameters_to_constant(self):
        for mod in self.stream_module.modules():
            if isinstance(mod, (torch.nn.Conv2d)):
                # to counter floating precision errors, we assign 1 to the weights and
                # normalize the output after the conv.
                torch.nn.init.constant_(mod.weight, 1)
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias, 0)

        for m in self.stream_module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                # Perhaps change to torch.nn.init.ones_(m.weight) and zeros?
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.eval()

    def _set_cudnn_flags(self, deterministic_flag, benchmark_flag):
        torch.backends.cudnn.deterministic = deterministic_flag
        torch.backends.cudnn.benchmark = benchmark_flag

    def _set_cudnn_flags_to_determistic(self):
        deterministic_flag = torch.backends.cudnn.deterministic
        benchmark_flag = torch.backends.cudnn.benchmark
        self._set_cudnn_flags(True, False)
        return deterministic_flag, benchmark_flag

    def _save_parameters(self):
        state_dict = self.stream_module.state_dict()
        state_dict = copy.deepcopy(state_dict)
        return state_dict

    def _restore_parameters(self, state_dict):
        self.stream_module.load_state_dict(state_dict)

    def disable(self):
        """Disable the streaming hooks"""
        self._remove_hooks()
        self.stream_module = self._reset_converted_modules(self.stream_module)

    def enable(self):
        """Enable the streaming hooks"""
        self._remove_hooks()
        self._streaming_reducers = []
        self.stream_module = self._convert_modules_for_streaming(self.stream_module)
        self._add_hooks_for_streaming()

    def __call__(self, image, **kwargs):
        result_on_cpu = kwargs.pop("result_on_cpu", False)
        return self.forward(image, result_on_cpu=result_on_cpu, **kwargs)
