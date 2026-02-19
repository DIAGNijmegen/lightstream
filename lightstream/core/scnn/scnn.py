"""
Author: Hans Pinckaers
MIT License
"""
import math
import copy
import contextlib
from typing import List

import numpy as np
import torch
import torch.autograd
import torch.backends
import torch.nn.functional

from lightstream.core.scnn.utils import Sides, Box, Lost, _ntuple, _new_value_indices, B_DIM, C_DIM, H_DIM, W_DIM
from lightstream.core.scnn.streamingconv import StreamingConv2d
from lightstream.core.scnn.streamingupsample import StreamingUpsample
from lightstream.core.scnn.streamingreducer import StreamingGlobalReducer
from lightstream.models.segment.reducer import GlobalReducer


_triple = _ntuple(3)
_STREAMING_MODULE_TYPES = (StreamingConv2d, StreamingUpsample, StreamingGlobalReducer)


class StreamingCNN(torch.nn.Module):
    """Initialize Streaming CNN helper class. After initialization use the
    forward() and backward() function of this class to lightstream.
    Pseudocode example:

    ```python
    sCNN = StreamingCNN(stream_layers, tile_shape=(1, 3, 600, 600))
    str_output = sCNN.forward(image)
    final_output = final_layers(str_output)
    loss = criterion(final_output, labels)
    loss.backward()
    sCNN.backward(image, str_output.grad)
    ```

    Hooks are used to perform lightstream, to use the stream_layers without
    lightstream you can disable StreamingCNN with the disable() function.
    Subsequently, enable() enables it again. Streaming gets enabled by default
    after initialization.
    """

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
            verbose (bool): will log various debugging relevant information (default is False)
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

        self._tile_output_shape = None
        self._module_stats = {}
        self._backward_seen_indices = {}
        self._saved_tensors = {}
        self._current_tile_input_loc = None
        self._hooks = []
        self._output_structure = None

        if state_dict is None:
            self._configure()
        else:
            self.load_tile_cache(state_dict)

    def _configure(self):
        # Save current model and cudnn flags, since we need to change them and restore later
        state_dict = self._save_parameters()
        (old_deterministic_flag, old_benchmark_flag) = self._set_cudnn_flags_to_determistic()
        self._reset_parameters_to_constant()

        # Add hooks to each layer to gather statistics
        self._add_hooks_for_statistics()

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
        if self.verbose:
            print("")
        self._gather_backward_statistics(tile)

        # TODO; temp hack for tile sizes too big on gpu,
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cuda()
            self.device = torch.device("cuda")  # type:ignore

        # Remove all hooks and add hooks for correcting gradients
        # during lightstream
        self._remove_hooks()
        #
        self._restore_parameters(state_dict)
        self._convert_modules_for_streaming(self.stream_module)
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

    def _gather_backward_statistics(self, tile):
        # Forward pass with grads enabled
        torch.set_grad_enabled(True)
        output = self.stream_module(tile)
        outputs, _ = self._split_outputs(output)

        # Gather backward statistics
        if len(outputs) == 1:
            self._tile_output_shape = outputs[0].shape
            gradient = torch.zeros(*outputs[0].shape, dtype=self.dtype, device=self.device)
            if outputs[0].ndim >= 4:
                gradient[
                    :,
                    :,
                    self.tile_output_lost.top : outputs[0].shape[H_DIM] - self.tile_output_lost.bottom,
                    self.tile_output_lost.left : outputs[0].shape[W_DIM] - self.tile_output_lost.right,
                ] = 1
            else:
                gradient.fill_(1)
            outputs[0].backward(gradient=gradient)
        else:
            self._tile_output_shape = [out.shape for out in outputs]
            gradients = []
            for out, lost in zip(outputs, self.tile_output_lost):
                gradient = torch.zeros(*out.shape, dtype=self.dtype, device=self.device)
                if out.ndim >= 4:
                    gradient[
                        :,
                        :,
                        lost.top : out.shape[H_DIM] - lost.bottom,
                        lost.left : out.shape[W_DIM] - lost.right,
                    ] = 1
                else:
                    gradient.fill_(1)
                gradients.append(gradient)

            self.tile_gradient_lost = []
            for idx, (out, gradient) in enumerate(zip(outputs, gradients)):
                tile_grad = torch.autograd.grad(
                    out,
                    tile,
                    grad_outputs=gradient,
                    # Keep the graph intact for the subsequent full backward call
                    # that populates module-level statistics.
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                self.tile_gradient_lost.append(self._non_max_border_amount(tile_grad))

            tile.grad = None
            torch.autograd.backward(outputs, grad_tensors=gradients)

        # Calculate the output stride of the whole stream_module
        if len(outputs) == 1:
            p_stats = self._prev_stats(outputs[0])
            if p_stats:
                self.output_stride = p_stats["output_stride"] * torch.tensor(p_stats["stride"])
            else:
                self.output_stride = torch.tensor([1, 1, 1])
        else:
            output_strides = []
            for out in outputs:
                p_stats = self._prev_stats(out)
                if p_stats:
                    output_stride = p_stats["output_stride"] * torch.tensor(p_stats["stride"])
                else:
                    output_stride = torch.tensor([1, 1, 1])
                output_strides.append(output_stride)
            self.output_stride = output_strides

        # tiles can have -1, see backward_statistics_hook
        if len(outputs) == 1:
            self.tile_gradient_lost = self._non_max_border_amount(tile.grad)

        # lost statistics assume you're always in the middle of an image, so left,bottom,top,right lost can always happen
        if self.verbose:
            print("\n", "Input gradient lost", self.tile_gradient_lost)

    def _gather_forward_statistics(self, tile):
        torch.set_grad_enabled(False)
        output = self.stream_module(tile)
        outputs, structure = self._split_outputs(output)
        self._output_structure = structure
        if len(outputs) == 1:
            self.tile_output_lost = self._non_max_border_amount(outputs[0])
        else:
            self.tile_output_lost = [self._non_max_border_amount(out) for out in outputs]
        if self.verbose:
            print("\n", "Output lost", self.tile_output_lost)

    def _convert_modules_for_streaming(self, module):
        mod = module
        if isinstance(module, torch.nn.Conv2d):
            if module in self._module_stats:
                mod = StreamingConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None,
                )
                mod = mod.to(module.weight.device, non_blocking=True)
                mod = mod.to(module.weight.dtype)

                mod.weight.requires_grad = module.weight.requires_grad
                if module.bias is not None:
                    mod.bias.requires_grad = module.bias.requires_grad

                mod.load_state_dict(module.state_dict())  # copy params
                mod.grad_lost = self._module_stats[module]["grad_lost"]
                mod.output_stride = self._module_stats[module]["output_stride"]
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, torch.nn.Upsample):
            if module in self._module_stats:
                mod = StreamingUpsample(
                    size=module.size,
                    scale_factor=module.scale_factor,
                    mode=module.mode,
                    align_corners=module.align_corners,
                )
                mod = mod.to(self.device, non_blocking=True)
                mod.grad_lost = self._module_stats[module]["grad_lost"]
                mod.output_stride = self._module_stats[module]["output_stride"]
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, GlobalReducer):
            mod = StreamingGlobalReducer(r=module.r, eps=module.eps)
            mod = mod.to(self.device, non_blocking=True)
            if module in self._module_stats:
                if "grad_lost" in self._module_stats[module]:
                    mod.grad_lost = self._module_stats[module]["grad_lost"]
                if "lost" in self._module_stats[module]:
                    mod.lost = self._module_stats[module]["lost"]
                mod.output_stride = self._module_stats[module]["output_stride"]
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        for name, child in module.named_children():
            mod.add_module(name, self._convert_modules_for_streaming(child))
        del module
        return mod

    def _reset_converted_modules(self, module):
        mod = module
        if isinstance(module, StreamingConv2d):
            mod = torch.nn.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
            )
            mod = mod.to(module.weight.device, non_blocking=True)
            mod = mod.to(module.weight.dtype)

            mod.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                mod.bias.requires_grad = module.bias.requires_grad

            mod.load_state_dict(module.state_dict())  # copy params
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, StreamingUpsample):
            mod = torch.nn.Upsample(
                size=module.size,
                scale_factor=module.scale_factor,
                mode=module.mode,
                align_corners=module.align_corners,
            )
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, StreamingGlobalReducer):
            mod = GlobalReducer(r=module.r, eps=module.eps)
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["lost"] = module.lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        for name, child in module.named_children():
            mod.add_module(name, self._reset_converted_modules(child))
        del module
        return mod

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

    def _non_max_border_amount(self, tensor):
        # Sum over the channels, useful for networks that treat certain channels
        # different (e.g., DenseNet)
        if tensor.dim() > 3:
            tensor = torch.sum(tensor, dim=1)[0]

        # Non-spatial tensors (e.g. [N, C] reducer outputs) do not have H/W borders.
        if tensor.dim() < 2:
            return Lost(0, 0, 0, 0)

        max_val = tensor.max()
        if not torch.isfinite(max_val) or float(max_val.abs().item()) <= self.eps:
            return Lost(0, 0, 0, 0)

        tensor = tensor / max_val  # normalize
        tensor = tensor > tensor.max() * (1 - self.eps)
        non_zero = tensor.nonzero(as_tuple=False)
        if non_zero.numel() == 0:
            return Lost(0, 0, 0, 0)

        top, left = non_zero.min(dim=0)[0]
        # for bottom and right we need to substract -1: correct index 3 is actually the 4th pixel
        bottom, right = (
            torch.tensor([*tensor.size()], dtype=torch.long, device=tensor.device) - non_zero.max(dim=0)[0] - 1
        )
        return Lost(int(top), int(left), int(bottom), int(right))

    def _split_outputs(self, output):
        if isinstance(output, dict):
            keys = list(output.keys())
            return [output[key] for key in keys], ("dict", keys)
        if isinstance(output, (list, tuple)):
            return list(output), ("sequence", type(output))
        return [output], None

    def _restore_outputs(self, outputs, structure):
        if structure is None:
            return outputs[0]
        kind, meta = structure
        if kind == "dict":
            return {key: value for key, value in zip(meta, outputs)}
        if kind == "sequence":
            if meta is tuple:
                return tuple(outputs)
            if isinstance(meta, type) and issubclass(meta, tuple):
                return meta(*outputs)
            return list(outputs)
        return outputs

    def _output_count(self):
        return len(self._tile_output_shape) if isinstance(self._tile_output_shape, list) else 1

    def _get_output_stride(self, index):
        if isinstance(self.output_stride, list):
            return self.output_stride[index]
        return self.output_stride

    def _stride_value(self, output_stride, index):
        if isinstance(output_stride, torch.Tensor):
            return float(output_stride[index].item())
        return float(output_stride[index])

    def _floor_div(self, numerator, denominator):
        return int(math.floor(float(numerator) / float(denominator)))

    def _mul_stride(self, value, stride):
        return int(round(float(value) * float(stride)))

    def _upsample_scale_factors(self, module, inpt, output):
        scale_factor = module.scale_factor
        if scale_factor is None:
            return (
                float(output.shape[H_DIM]) / float(inpt[0].shape[H_DIM]),
                float(output.shape[W_DIM]) / float(inpt[0].shape[W_DIM]),
            )
        if isinstance(scale_factor, (tuple, list)):
            if len(scale_factor) == 1:
                return float(scale_factor[0]), float(scale_factor[0])
            return float(scale_factor[0]), float(scale_factor[1])
        return float(scale_factor), float(scale_factor)


    def _crop_spatial_for_sides(self, tensor, target_h, target_w, sides):
        h, w = tensor.shape[H_DIM], tensor.shape[W_DIM]

        if h > target_h:
            if sides.top and not sides.bottom:
                tensor = tensor[:, :, :target_h, :]
            elif sides.bottom and not sides.top:
                tensor = tensor[:, :, h - target_h :, :]
            else:
                start = (h - target_h) // 2
                tensor = tensor[:, :, start : start + target_h, :]

        if w > target_w:
            if sides.left and not sides.right:
                tensor = tensor[:, :, :, :target_w]
            elif sides.right and not sides.left:
                tensor = tensor[:, :, :, w - target_w :]
            else:
                start = (w - target_w) // 2
                tensor = tensor[:, :, :, start : start + target_w]

        return tensor

    def _align_trimmed_tensors(self, trimmed_grad, trimmed_output, sides):
        target_h = min(trimmed_grad.shape[H_DIM], trimmed_output.shape[H_DIM])
        target_w = min(trimmed_grad.shape[W_DIM], trimmed_output.shape[W_DIM])

        trimmed_grad = self._crop_spatial_for_sides(trimmed_grad, target_h, target_w, sides)
        trimmed_output = self._crop_spatial_for_sides(trimmed_output, target_h, target_w, sides)

        return trimmed_grad, trimmed_output

    def _get_tile_output_shape(self, index):
        if isinstance(self._tile_output_shape, list):
            return self._tile_output_shape[index]
        return self._tile_output_shape

    def _get_tile_output_lost(self, index):
        if isinstance(self.tile_output_lost, list):
            return self.tile_output_lost[index]
        return self.tile_output_lost

    def _get_tile_gradient_lost(self, index):
        if isinstance(self.tile_gradient_lost, list):
            return self.tile_gradient_lost[index]
        return self.tile_gradient_lost

    def _outputs_share_tiling(self):
        if self._output_count() == 1:
            return True
        shapes = self._tile_output_shape
        losts = self.tile_output_lost
        strides = self.output_stride

        gradient_losts = None
        if isinstance(self.tile_gradient_lost, list):
            gradient_losts = self.tile_gradient_lost

        for idx in range(1, self._output_count()):
            if shapes[idx] != shapes[0]:
                return False
            if losts[idx] != losts[0]:
                return False
            if not torch.equal(strides[idx], strides[0]):
                return False
            if gradient_losts is not None and gradient_losts[idx] != gradient_losts[0]:
                return False
        return True

    def _tiling_signature(self, output_index):
        tile_shape = self._get_tile_output_shape(output_index)
        tile_lost = self._get_tile_output_lost(output_index)
        tile_grad_lost = self._get_tile_gradient_lost(output_index)
        stride = self._get_output_stride(output_index)

        if isinstance(stride, torch.Tensor):
            stride = tuple(float(v) for v in stride.tolist())
        else:
            stride = tuple(float(v) for v in stride)

        return tile_shape, tile_lost, tile_grad_lost, stride

    def _output_tiling_groups(self):
        groups = {}
        for output_index in range(self._output_count()):
            signature = self._tiling_signature(output_index)
            groups.setdefault(signature, []).append(output_index)
        return list(groups.values())

    def _output_overlap_groups(self):
        groups = {}
        for output_index in range(self._output_count()):
            output_shape = self._get_tile_output_shape(output_index)
            output_stride = self._get_output_stride(output_index)

            if isinstance(output_stride, torch.Tensor):
                stride_signature = tuple(float(v) for v in output_stride.tolist())
            else:
                stride_signature = tuple(float(v) for v in output_stride)

            groups.setdefault((output_shape, stride_signature), []).append(output_index)

        return list(groups.values())

    def _planning_output_index(self, output_index):
        shape = self._get_tile_output_shape(output_index)
        if len(shape) >= 4:
            return output_index

        # Debug/model-specific pairing: reducer heads often correspond to
        # later full-resolution map outputs (e.g., [0:3] with [4:7]).
        if output_index + 4 < self._output_count():
            paired_shape = self._get_tile_output_shape(output_index + 4)
            if len(paired_shape) >= 4:
                return output_index + 4

        # Fallback: first spatial output.
        for idx in range(self._output_count()):
            if len(self._get_tile_output_shape(idx)) >= 4:
                return idx
        return output_index

    def _forward_single_output(self, image, result_on_cpu, output_index=0, initialize_saliency=True):
        tile_width, tile_height = self.tile_shape[W_DIM], self.tile_shape[H_DIM]
        planning_index = self._planning_output_index(output_index)
        tile_output_shape = self._get_tile_output_shape(planning_index)
        tile_output_lost = self._get_tile_output_lost(planning_index)
        output_stride = self._get_output_stride(planning_index)
        stride_y = self._stride_value(output_stride, 1)
        stride_x = self._stride_value(output_stride, 2)

        # Size of valid output of a tile
        valid_output_height = tile_output_shape[H_DIM] - tile_output_lost.top - tile_output_lost.bottom
        valid_output_width = tile_output_shape[W_DIM] - tile_output_lost.left - tile_output_lost.right

        # We will keep track which part of the output of the whole image we
        # already filled with valid values from tile output.
        already_filled = Box(0, 0, 0, 0, None)

        # Calculate size of output that we would get by inferencing the
        # whole image.
        output_height = self._floor_div(image.shape[H_DIM] - self.tile_shape[H_DIM], stride_y) + tile_output_shape[H_DIM]
        output_width = self._floor_div(image.shape[W_DIM] - self.tile_shape[W_DIM], stride_x) + tile_output_shape[W_DIM]

        if result_on_cpu:
            device = torch.device("cpu")
        else:
            device = self.device
        output = torch.empty(
            (image.shape[0], tile_output_shape[1], output_height, output_width), dtype=self.dtype, device=device
        ).fill_(999)

        n_rows = math.ceil(float(output_height) / float(valid_output_height))
        n_cols = math.ceil(float(output_width) / float(valid_output_width))

        if image.shape[W_DIM] <= tile_width:
            n_cols = 1
        if image.shape[H_DIM] <= tile_height:
            n_rows = 1

        if self.gather_input_gradient and initialize_saliency:
            self.saliency_map = torch.zeros(image.shape, dtype=self.dtype, device="cpu")

        # if self.verbose:
        #    print("Number of tiles in forward:", n_rows * n_cols)
        # if self.verbose:
        #    iterator = tqdm(range(n_rows))
        # else:
        iterator = range(n_rows)
        relevant_output = None

        with torch.no_grad():
            for row in iterator:
                for col in range(n_cols):
                    # Coordinates of the output w.r.t. the output of full image
                    output_y = row * valid_output_height
                    output_x = col * valid_output_width

                    # Check if we are at borders, since we can not create
                    # overlap here and should not crop values.
                    sides_top = True if row == 0 else False
                    sides_left = True if col == 0 else False

                    sides_bottom = True if output_y * stride_y + self.tile_shape[H_DIM] >= image.shape[H_DIM] else False
                    sides_right = True if output_x * stride_x + self.tile_shape[W_DIM] >= image.shape[W_DIM] else False
                    sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                    # These values are used to crop invalid output values.
                    # For non-spatial outputs (e.g. reducer heads), use the
                    # paired spatial planning index so reducer tile coordinates
                    # align with map reconstruction geometry.
                    lost = self._get_tile_lost_for_sides(sides, planning_index)

                    # Since we need to stay at multiples of output stride we
                    # need to keep that into account when we are at the bottom
                    # and right side of the output.
                    if sides_bottom:
                        output_y = self._floor_div(image.shape[H_DIM] - self.tile_shape[H_DIM], stride_y)
                    if sides_right:
                        output_x = self._floor_div(image.shape[W_DIM] - self.tile_shape[W_DIM], stride_x)

                    output_y = output_y if not sides.top else 0
                    output_x = output_x if not sides.left else 0
                    output_loc = Box(output_y + lost.top, -1, output_x + lost.left, -1, sides)

                    # Coordinates of the input w.r.t. the output of full image
                    tile_y = self._mul_stride(output_y, stride_y)
                    tile_x = self._mul_stride(output_x, stride_x)

                    # Extract tile and perform forward pass
                    tile = image[:, :, tile_y : tile_y + tile_height, tile_x : tile_x + tile_width]

                    # normalize on gpu for speed in dataloader
                    # does this reduce speed significantly?
                    if not self.copy_to_gpu:
                        tile = tile.to(self.device, non_blocking=True)

                    if self.should_normalize:
                        tile = self._normalize_on_gpu(tile)

                    input_loc = Box(tile_y, tile_height, tile_x, tile_width, sides)
                    for mod in self.stream_module.modules():
                        if isinstance(mod, _STREAMING_MODULE_TYPES):
                            mod.input_loc = input_loc
                            if isinstance(mod, StreamingGlobalReducer):
                                mod.lost = self._get_tile_output_lost(planning_index)
                                mod.output_stride = self._get_output_stride(planning_index)
                                mod.data_loc = Box(output_y + lost.top, 0, output_x + lost.left, 0, sides)

                    tile_output = self.stream_module(tile)
                    outputs, _ = self._split_outputs(tile_output)
                    if output_index >= len(outputs):
                        raise ValueError("Streaming output index is out of range for the model outputs.")
                    tile_output = outputs[output_index]

                    if torch.backends.cudnn.benchmark:
                        torch.cuda.empty_cache()

                    if tile_output.ndim < 4:
                        output = tile_output.to(device, non_blocking=True)
                        del tile
                        continue

                    trimmed_output = tile_output[
                        :,
                        :,
                        lost.top : tile_output.shape[H_DIM] - lost.bottom,
                        lost.left : tile_output.shape[W_DIM] - lost.right,
                    ]

                    new_output_box, updated_total_indices = _new_value_indices(
                        trimmed_output.shape, output_loc, already_filled
                    )
                    already_filled = updated_total_indices

                    relevant_output = trimmed_output[
                        :,
                        :,
                        new_output_box.y : updated_total_indices.y + new_output_box.height,
                        new_output_box.x : new_output_box.x + new_output_box.width,
                    ]

                    output[
                        :,
                        :,
                        int(updated_total_indices.y) : int(updated_total_indices.height),
                        int(updated_total_indices.x - new_output_box.width) : int(updated_total_indices.x),
                    ] = relevant_output

                    del tile

            assert sides_bottom and sides_right, "It seems like we could not reconstruct all output"  # type:ignore

        # mem management
        if relevant_output is not None:
            del relevant_output
        self._saved_tensors = {}

        return output

    def _forward_multi_output_shared(self, image, result_on_cpu):
        outputs = []
        already_filled = []
        tile_output_shape = self._get_tile_output_shape(0)
        tile_output_lost = self._get_tile_output_lost(0)
        output_stride = self._get_output_stride(0)
        stride_y = self._stride_value(output_stride, 1)
        stride_x = self._stride_value(output_stride, 2)

        tile_width, tile_height = self.tile_shape[W_DIM], self.tile_shape[H_DIM]

        valid_output_height = tile_output_shape[H_DIM] - tile_output_lost.top - tile_output_lost.bottom
        valid_output_width = tile_output_shape[W_DIM] - tile_output_lost.left - tile_output_lost.right

        output_height = self._floor_div(image.shape[H_DIM] - self.tile_shape[H_DIM], stride_y) + tile_output_shape[H_DIM]
        output_width = self._floor_div(image.shape[W_DIM] - self.tile_shape[W_DIM], stride_x) + tile_output_shape[W_DIM]

        if result_on_cpu:
            device = torch.device("cpu")
        else:
            device = self.device

        for _ in range(self._output_count()):
            output = torch.empty(
                (image.shape[0], tile_output_shape[1], output_height, output_width), dtype=self.dtype, device=device
            ).fill_(999)
            outputs.append(output)
            already_filled.append(Box(0, 0, 0, 0, None))

        n_rows = math.ceil(float(output_height) / float(valid_output_height))
        n_cols = math.ceil(float(output_width) / float(valid_output_width))

        if image.shape[W_DIM] <= tile_width:
            n_cols = 1
        if image.shape[H_DIM] <= tile_height:
            n_rows = 1

        if self.gather_input_gradient:
            self.saliency_map = torch.zeros(image.shape, dtype=self.dtype, device="cpu")

        iterator = range(n_rows)
        relevant_output = None

        with torch.no_grad():
            for row in iterator:
                for col in range(n_cols):
                    output_y = row * valid_output_height
                    output_x = col * valid_output_width

                    sides_top = True if row == 0 else False
                    sides_left = True if col == 0 else False

                    sides_bottom = True if output_y * stride_y + self.tile_shape[H_DIM] >= image.shape[H_DIM] else False
                    sides_right = True if output_x * stride_x + self.tile_shape[W_DIM] >= image.shape[W_DIM] else False
                    sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                    lost = self._get_tile_lost_for_sides(sides, 0)

                    if sides_bottom:
                        output_y = self._floor_div(image.shape[H_DIM] - self.tile_shape[H_DIM], stride_y)
                    if sides_right:
                        output_x = self._floor_div(image.shape[W_DIM] - self.tile_shape[W_DIM], stride_x)

                    output_y = output_y if not sides.top else 0
                    output_x = output_x if not sides.left else 0
                    output_loc = Box(output_y + lost.top, -1, output_x + lost.left, -1, sides)

                    tile_y = self._mul_stride(output_y, stride_y)
                    tile_x = self._mul_stride(output_x, stride_x)

                    tile = image[:, :, tile_y : tile_y + tile_height, tile_x : tile_x + tile_width]

                    if not self.copy_to_gpu:
                        tile = tile.to(self.device, non_blocking=True)

                    if self.should_normalize:
                        tile = self._normalize_on_gpu(tile)

                    input_loc = Box(tile_y, tile_height, tile_x, tile_width, sides)
                    for mod in self.stream_module.modules():
                        if isinstance(mod, _STREAMING_MODULE_TYPES):
                            mod.input_loc = input_loc

                    tile_output = self.stream_module(tile)
                    tile_outputs, _ = self._split_outputs(tile_output)
                    if len(tile_outputs) != self._output_count():
                        raise ValueError("Streaming output count does not match model outputs.")

                    if torch.backends.cudnn.benchmark:
                        torch.cuda.empty_cache()

                    for output_index, output_tensor in enumerate(outputs):
                        tile_output = tile_outputs[output_index]
                        if tile_output.ndim < 4:
                            outputs[output_index] = tile_output.to(device, non_blocking=True)
                            continue

                        trimmed_output = tile_output[
                            :,
                            :,
                            lost.top : tile_output.shape[H_DIM] - lost.bottom,
                            lost.left : tile_output.shape[W_DIM] - lost.right,
                        ]

                        new_output_box, updated_total_indices = _new_value_indices(
                            trimmed_output.shape, output_loc, already_filled[output_index]
                        )
                        already_filled[output_index] = updated_total_indices

                        relevant_output = trimmed_output[
                            :,
                            :,
                            new_output_box.y : updated_total_indices.y + new_output_box.height,
                            new_output_box.x : new_output_box.x + new_output_box.width,
                        ]

                        output_tensor[
                            :,
                            :,
                            int(updated_total_indices.y) : int(updated_total_indices.height),
                            int(updated_total_indices.x - new_output_box.width) : int(updated_total_indices.x),
                        ] = relevant_output

                    del tile

            assert sides_bottom and sides_right, "It seems like we could not reconstruct all output"  # type:ignore

        if relevant_output is not None:
            del relevant_output
        self._saved_tensors = {}

        return outputs

    def forward(self, image, result_on_cpu=False):
        """Perform forward pass with lightstream.

        Parameters:
            image (torch.Tensor): CHW the image to lightstream
        """
        image = image

        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)

        for mod in self.stream_module.modules():
            if isinstance(mod, _STREAMING_MODULE_TYPES):
                mod.reset()
                mod.input_loc = None
                if isinstance(mod, StreamingGlobalReducer):
                    mod.data_loc = None

        if self._output_count() == 1:
            output = self._forward_single_output(image, result_on_cpu)
            del image
            return output

        if self._outputs_share_tiling():
            outputs = self._forward_multi_output_shared(image, result_on_cpu)
        else:
            outputs = []
            for idx in range(self._output_count()):
                if idx > 0:
                    for mod in self.stream_module.modules():
                        if isinstance(mod, _STREAMING_MODULE_TYPES):
                            mod.reset()
                            mod.input_loc = None
                            if isinstance(mod, StreamingGlobalReducer):
                                mod.data_loc = None
                outputs.append(
                    self._forward_single_output(image, result_on_cpu, output_index=idx, initialize_saliency=(idx == 0))
                )

        for mod in self.stream_module.modules():
            if isinstance(mod, _STREAMING_MODULE_TYPES):
                mod.input_loc = None
                if isinstance(mod, StreamingGlobalReducer):
                    mod.data_loc = None

        del image
        return self._restore_outputs(outputs, self._output_structure)


    def _streaming_reducer_for_output(self, output_index):
        reducers = [m for m in self.stream_module.modules() if isinstance(m, StreamingGlobalReducer)]
        if output_index < len(reducers):
            return reducers[output_index]
        return None

    def _reducer_grad_to_spatial(self, spatial_output, reducer_grad, reducer_module):
        if reducer_module is None:
            return None

        r = float(reducer_module.r)
        eps = float(reducer_module.eps)

        probs = torch.sigmoid(spatial_output)
        p_r = probs.pow(r)
        mean_p_r_raw = p_r.mean(dim=(-2, -1), keepdim=True)
        mean_p_r = mean_p_r_raw.clamp_min(eps)

        # d/dx clamp_min(x, eps) = 0 when x < eps, 1 otherwise.
        clamp_mask = (mean_p_r_raw >= eps).to(mean_p_r.dtype)

        # Match denominator used by StreamingGlobalReducer forward accumulation.
        reducer_count = getattr(reducer_module, "_count", None)
        if reducer_count is None or int(reducer_count) <= 0:
            numel = float(spatial_output.shape[H_DIM] * spatial_output.shape[W_DIM])
        else:
            numel = float(int(reducer_count))

        scale = (mean_p_r.pow((1.0 / r) - 1.0) * clamp_mask) / numel

        spatial_grad = reducer_grad[:, :, None, None] * scale * probs.pow(r - 1.0) * probs * (1.0 - probs)
        return spatial_grad

    def _backward_single_output(self, image, grad, output_index=0):
        grad = grad

        output_shape = self._get_tile_output_shape(output_index)
        if len(output_shape) < 4:
            planning_index = self._planning_output_index(output_index)
            if planning_index == output_index:
                raise ValueError("Non-spatial output does not have a paired spatial planning output for backward.")

            # For reducer outputs, use paired spatial replay to preserve
            # per-tile immediate backward order required by streaming ops.
            # Chaining all tiles into one backward pass triggers reverse-order
            # traversal and breaks seen-index assumptions in streaming upsample/conv.
            for mod in self.stream_module.modules():
                if isinstance(mod, _STREAMING_MODULE_TYPES):
                    mod.reset()
                    mod.input_loc = None
                    if isinstance(mod, StreamingGlobalReducer):
                        mod.data_loc = None

            with torch.no_grad():
                spatial_output = self._forward_single_output(
                    image,
                    result_on_cpu=False,
                    output_index=planning_index,
                    initialize_saliency=False,
                )

            reducer_module = self._streaming_reducer_for_output(output_index)
            spatial_grad = self._reducer_grad_to_spatial(spatial_output, grad, reducer_module)
            if spatial_grad is None:
                raise ValueError("Could not map reducer output gradient to spatial gradient.")

            for mod in self.stream_module.modules():
                if isinstance(mod, _STREAMING_MODULE_TYPES):
                    mod.reset()
                    mod.input_loc = None
                    if isinstance(mod, StreamingGlobalReducer):
                        mod.data_loc = None

            return self._backward_single_output(image, spatial_grad, output_index=planning_index)

        height = image.shape[H_DIM]
        width = image.shape[W_DIM]

        tile_height = self.tile_shape[H_DIM]
        tile_width = self.tile_shape[W_DIM]
        grad_lost = self._get_tile_gradient_lost(output_index)

        output_stride = self._get_output_stride(output_index)
        stride_y = self._stride_value(output_stride, 1)
        stride_x = self._stride_value(output_stride, 2)
        output_shape = self._get_tile_output_shape(output_index)

        output_height = output_shape[H_DIM]
        output_width = output_shape[W_DIM]

        valid_grad_height = math.floor((tile_height - grad_lost.top - grad_lost.bottom) / stride_y) * stride_y
        valid_grad_width = math.floor((tile_width - grad_lost.left - grad_lost.right) / stride_x) * stride_x

        n_rows = math.ceil(float(height - grad_lost.top - grad_lost.bottom) / float(valid_grad_height))
        n_cols = math.ceil(float(width - grad_lost.left - grad_lost.right) / float(valid_grad_width))

        if image.shape[W_DIM] <= tile_width:
            n_cols = 1
        if image.shape[H_DIM] <= tile_height:
            n_rows = 1

        self._inputs = {}
        self._backward_seen_indices = {}

        iterator = range(n_rows)

        for row in iterator:
            for col in range(n_cols):
                output_y = self._floor_div(row * valid_grad_height, stride_y)
                output_x = self._floor_div(col * valid_grad_width, stride_x)

                sides_top = True if row == 0 else False
                sides_left = True if col == 0 else False

                sides_bottom = True if output_y + output_height >= grad.shape[H_DIM] else False
                sides_right = True if output_x + output_width >= grad.shape[W_DIM] else False
                sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                lost = self._get_tile_lost_for_sides(sides, output_index)

                if sides_bottom:
                    output_y = max(grad.shape[H_DIM] - output_height, 0)
                if sides_right:
                    output_x = max(grad.shape[W_DIM] - output_width, 0)

                input_y = self._mul_stride(output_y, stride_y)
                input_x = self._mul_stride(output_x, stride_x)

                input_loc = Box(input_y, tile_height, input_x, tile_width, sides)

                tile = image[:, :, input_y : input_y + tile_height, input_x : input_x + tile_width]

                gradient = grad[:, :, output_y : output_y + output_height, output_x : output_x + output_width]

                self._saved_tensors = {}

                trimmed_grad = gradient[
                    :, :, lost.top : gradient.shape[H_DIM] - lost.bottom, lost.left : gradient.shape[W_DIM] - lost.right
                ]

                if not self.copy_to_gpu:
                    tile = tile.to(self.device, non_blocking=True)

                for mod in self.stream_module.modules():
                    if isinstance(mod, _STREAMING_MODULE_TYPES):
                        mod.input_loc = input_loc

                if self.should_normalize:
                    tile = self._normalize_on_gpu(tile)

                if self.gather_input_gradient:
                    tile.requires_grad = True
                    self.saliency_old_indices = copy.deepcopy(self.saliency_input_module.seen_indices)

                autocast_enabled = tile.is_cuda and self.dtype in (torch.float16, torch.bfloat16)
                autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if autocast_enabled else contextlib.nullcontext()
                with autocast_ctx:
                    tile_output = self.stream_module(tile)

                del tile

                outputs, _ = self._split_outputs(tile_output)
                if output_index >= len(outputs):
                    raise ValueError("Streaming output index is out of range for the model outputs.")
                tile_output = outputs[output_index]

                trimmed_output = tile_output[
                    :,
                    :,
                    lost.top : tile_output.shape[H_DIM] - lost.bottom,
                    lost.left : tile_output.shape[W_DIM] - lost.right,
                ]

                trimmed_output = trimmed_output.to(self.device, non_blocking=True)

                if (
                    trimmed_grad.shape[H_DIM] != trimmed_output.shape[H_DIM]
                    or trimmed_grad.shape[W_DIM] != trimmed_output.shape[W_DIM]
                ):
                    trimmed_grad, trimmed_output = self._align_trimmed_tensors(trimmed_grad, trimmed_output, sides)

                trimmed_output.backward(trimmed_grad)

                del tile_output
                del trimmed_grad
                del trimmed_output

    def _backward_multi_output_shared(self, image, grads, output_indices=None, grad_lost=None):
        if output_indices is None:
            output_indices = list(range(len(grads)))
        if len(output_indices) == 0:
            return

        reference_output = output_indices[0]
        height = image.shape[H_DIM]
        width = image.shape[W_DIM]

        tile_height = self.tile_shape[H_DIM]
        tile_width = self.tile_shape[W_DIM]
        if grad_lost is None:
            grad_lost = self._get_tile_gradient_lost(reference_output)

        output_stride = self._get_output_stride(reference_output)
        stride_y = self._stride_value(output_stride, 1)
        stride_x = self._stride_value(output_stride, 2)
        output_shape = self._get_tile_output_shape(reference_output)

        output_height = output_shape[H_DIM]
        output_width = output_shape[W_DIM]

        valid_grad_height = math.floor((tile_height - grad_lost.top - grad_lost.bottom) / stride_y) * stride_y
        valid_grad_width = math.floor((tile_width - grad_lost.left - grad_lost.right) / stride_x) * stride_x

        n_rows = math.ceil(float(height - grad_lost.top - grad_lost.bottom) / float(valid_grad_height))
        n_cols = math.ceil(float(width - grad_lost.left - grad_lost.right) / float(valid_grad_width))

        if image.shape[W_DIM] <= tile_width:
            n_cols = 1
        if image.shape[H_DIM] <= tile_height:
            n_rows = 1

        self._inputs = {}
        self._backward_seen_indices = {}

        iterator = range(n_rows)

        for row in iterator:
            for col in range(n_cols):
                output_y = self._floor_div(row * valid_grad_height, stride_y)
                output_x = self._floor_div(col * valid_grad_width, stride_x)

                sides_top = True if row == 0 else False
                sides_left = True if col == 0 else False

                sides_bottom = True if output_y + output_height >= grads[reference_output].shape[H_DIM] else False
                sides_right = True if output_x + output_width >= grads[reference_output].shape[W_DIM] else False
                sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                lost = self._get_tile_lost_for_sides(sides, reference_output)

                if sides_bottom:
                    output_y = max(grads[reference_output].shape[H_DIM] - output_height, 0)
                if sides_right:
                    output_x = max(grads[reference_output].shape[W_DIM] - output_width, 0)

                input_y = self._mul_stride(output_y, stride_y)
                input_x = self._mul_stride(output_x, stride_x)

                input_loc = Box(input_y, tile_height, input_x, tile_width, sides)

                tile = image[:, :, input_y : input_y + tile_height, input_x : input_x + tile_width]

                self._saved_tensors = {}

                if not self.copy_to_gpu:
                    tile = tile.to(self.device, non_blocking=True)

                for mod in self.stream_module.modules():
                    if isinstance(mod, _STREAMING_MODULE_TYPES):
                        mod.input_loc = input_loc

                if self.should_normalize:
                    tile = self._normalize_on_gpu(tile)

                if self.gather_input_gradient:
                    tile.requires_grad = True
                    self.saliency_old_indices = copy.deepcopy(self.saliency_input_module.seen_indices)

                autocast_enabled = tile.is_cuda and self.dtype in (torch.float16, torch.bfloat16)
                autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if autocast_enabled else contextlib.nullcontext()
                with autocast_ctx:
                    tile_output = self.stream_module(tile)

                del tile

                outputs, _ = self._split_outputs(tile_output)
                if len(outputs) != self._output_count():
                    raise ValueError("Streaming output count does not match model outputs.")

                trimmed_outputs = []
                trimmed_grads = []
                for output_index in output_indices:
                    gradient = grads[output_index]
                    output_lost = self._get_tile_lost_for_sides(sides, output_index)
                    grad_tile = gradient[:, :, output_y : output_y + output_height, output_x : output_x + output_width]
                    trimmed_grad = grad_tile[
                        :,
                        :,
                        output_lost.top : grad_tile.shape[H_DIM] - output_lost.bottom,
                        output_lost.left : grad_tile.shape[W_DIM] - output_lost.right,
                    ]

                    tile_out = outputs[output_index]
                    trimmed_output = tile_out[
                        :,
                        :,
                        output_lost.top : tile_out.shape[H_DIM] - output_lost.bottom,
                        output_lost.left : tile_out.shape[W_DIM] - output_lost.right,
                    ]
                    trimmed_output = trimmed_output.to(self.device, non_blocking=True)

                    if (
                        trimmed_grad.shape[H_DIM] != trimmed_output.shape[H_DIM]
                        or trimmed_grad.shape[W_DIM] != trimmed_output.shape[W_DIM]
                    ):
                        trimmed_grad, trimmed_output = self._align_trimmed_tensors(trimmed_grad, trimmed_output, sides)

                    trimmed_outputs.append(trimmed_output)
                    trimmed_grads.append(trimmed_grad)

                torch.autograd.backward(trimmed_outputs, grad_tensors=trimmed_grads)

                del tile_output
                del trimmed_grads
                del trimmed_outputs

    def backward(self, image, grad):
        """Perform backward pass with lightstream.

        Parameters:
            image (torch.Tensor): the image (expects NCHW) that was used in the forward pass
            grad (torch.Tensor): this should be the gradient of the output of
                the stream_layers, or a structure matching the forward outputs.
        """
        image = image
        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)

        grads, grad_structure = self._split_outputs(grad)

        if len(grads) != self._output_count():
            raise ValueError("Gradient outputs do not match the number of model outputs.")

        if self._output_count() == 1:
            self._backward_single_output(image, grads[0])
        elif grad_structure == self._output_structure:
            has_streaming_reducer = any(isinstance(mod, StreamingGlobalReducer) for mod in self.stream_module.modules())

            for overlap_group in self._output_overlap_groups():
                # Shared backward currently supports only spatial outputs and can
                # introduce parity issues when reducer outputs are present.
                # In reducer-enabled models prefer per-output replay for correctness.
                has_non_spatial = any(len(self._get_tile_output_shape(idx)) < 4 for idx in overlap_group)
                if len(overlap_group) == 1 or has_non_spatial or has_streaming_reducer:
                    for idx in overlap_group:
                        grad_tensor = grads[idx]
                        if grad_tensor is None:
                            continue
                        if torch.count_nonzero(grad_tensor).item() == 0:
                            continue

                        # Each output branch should start from a fresh streaming state.
                        for mod in self.stream_module.modules():
                            if isinstance(mod, _STREAMING_MODULE_TYPES):
                                mod.reset()

                        self._backward_single_output(image, grad_tensor, output_index=idx)
                    continue

                grad_losts = [self._get_tile_gradient_lost(output_index) for output_index in overlap_group]
                max_grad_lost = Lost(
                    max(lost.top for lost in grad_losts),
                    max(lost.left for lost in grad_losts),
                    max(lost.bottom for lost in grad_losts),
                    max(lost.right for lost in grad_losts),
                )
                self._backward_multi_output_shared(
                    image,
                    grads,
                    output_indices=overlap_group,
                    grad_lost=max_grad_lost,
                )
        else:
            for idx, grad_tensor in enumerate(grads):
                if grad_tensor is None:
                    continue
                if torch.count_nonzero(grad_tensor).item() == 0:
                    continue

                # Each output branch should start from a fresh streaming state.
                # Otherwise, zero/other-output passes can advance seen_indices and
                # under-count gradients for the current output.
                for mod in self.stream_module.modules():
                    if isinstance(mod, _STREAMING_MODULE_TYPES):
                        mod.reset()

                self._backward_single_output(image, grad_tensor, output_index=idx)

        self._saved_tensors = {}
        self._current_tile_input_loc = None

        for mod in self.stream_module.modules():
            if isinstance(mod, _STREAMING_MODULE_TYPES):
                mod.input_loc = None
                if isinstance(mod, StreamingGlobalReducer):
                    mod.data_loc = None
                mod.reset()

        del image

    def _get_tile_lost_for_sides(self, sides, output_index=0):
        tile_output_lost = self._get_tile_output_lost(output_index)
        lost_top = tile_output_lost.top if not sides.top else 0
        lost_bottom = tile_output_lost.bottom if not sides.bottom else 0
        lost_left = tile_output_lost.left if not sides.left else 0
        lost_right = tile_output_lost.right if not sides.right else 0
        lost = Lost(lost_top, lost_left, lost_bottom, lost_right)
        return lost

    def _normalize_on_gpu(self, tile):
        tile_norm = tile.to(self.dtype)
        del tile
        tile_norm.div_(255)
        tile_norm.sub_(self.mean)
        tile_norm.div_(self.std)
        tile = tile_norm
        return tile

    def _normalize_stream_input(self, inpt):
        if not inpt:
            return inpt
        data = inpt[0]
        if isinstance(data, dict):
            if not data:
                return inpt
            data = next(iter(data.values()))
        elif isinstance(data, (list, tuple)):
            if not data:
                return inpt
            data = data[0]
        if data is inpt[0]:
            return inpt
        return (data,)

    def disable(self):
        """Disable the streaming hooks"""
        self._remove_hooks()
        self._reset_converted_modules(self.stream_module)

    def enable(self):
        """Enable the streaming hooks"""
        self._remove_hooks()
        self._convert_modules_for_streaming(self.stream_module)
        self._add_hooks_for_streaming()

    def _add_hooks_for_statistics(self):
        def pre_lambda(module, inpt):
            return self._normalize_stream_input(inpt)

        def forw_lambda(module, inpt, outpt):
            self._forward_gather_statistics_hook(module, inpt, outpt)

        def back_lambda(module, grad_in, grad_out):
            return self._backward_gather_statistics_hook(module, grad_in, grad_out)

        self._add_hooks(
            forward_pre_hook=pre_lambda,
            forward_hook=forw_lambda,
            backward_hook=back_lambda,
            forward_modules=(
                torch.nn.Conv2d,
                torch.nn.MaxPool2d,
                torch.nn.AvgPool2d,
                torch.nn.Upsample,
                StreamingUpsample,
                GlobalReducer,
            ),
            back_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Upsample, StreamingUpsample),
        )

    def _add_hooks_for_streaming(self):
        if self.gather_input_gradient:

            def back_lambda(module, grad_in, grad_out):
                return self._backward_saliency_hook(module, grad_in, grad_out)

            for mod in self.stream_module.modules():
                if isinstance(mod, StreamingConv2d):
                    if mod.in_channels == 3:
                        self.saliency_input_module = mod
                        back_handle = mod.register_full_backward_hook(back_lambda)
                        self._hooks.append(back_handle)

    def _add_hooks(
        self,
        forward_hook,
        backward_hook,
        forward_pre_hook=None,
        forward_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d),
        back_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d),
    ):
        for mod in self.stream_module.modules():
            if isinstance(mod, forward_modules):
                if forward_pre_hook:
                    pre_handle = mod.register_forward_pre_hook(forward_pre_hook)
                    self._hooks.append(pre_handle)
                forw_handle = mod.register_forward_hook(forward_hook)
                self._hooks.append(forw_handle)
                if back_modules and isinstance(mod, back_modules):
                    back_handle = mod.register_full_backward_hook(backward_hook)
                    self._hooks.append(back_handle)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def _forward_gather_statistics_hook(self, module, inpt, output):
        if isinstance(module, (GlobalReducer, StreamingGlobalReducer)):
            stride = (1.0, 1.0, 1.0)
            kernel_size = stride

            if not torch.is_grad_enabled():  # type:ignore
                prev_stats = self._prev_stats(inpt[0]) if len(inpt) > 0 else None
                if prev_stats and "lost" in prev_stats:
                    lost = prev_stats["lost"]
                else:
                    lost = self._non_max_border_amount(inpt[0])

                module_stats = {"lost": lost, "stride": stride, "module": module}
                if self.verbose:
                    print(module, "\n", module_stats["lost"])

                self._saved_tensors[module] = inpt
                self._module_stats[module] = module_stats
            else:
                module_stats = self._module_stats[module]

                p_stats = self._prev_stats(output)
                if p_stats:
                    output_stride = p_stats["output_stride"] * torch.tensor(p_stats["stride"])
                else:
                    output_stride = torch.tensor([1, 1, 1])

                module_stats["output_stride"] = output_stride.clone().detach()
                self._stats_per_grad_fn[output.grad_fn] = module_stats
                self._module_stats[module] = module_stats
            return

        if isinstance(module, (torch.nn.Upsample, StreamingUpsample)):
            if module.mode != "bilinear":
                raise ValueError("Streaming statistics only support bilinear upsample.")
            scale_y, scale_x = self._upsample_scale_factors(module, inpt, output)
            stride = (1.0, 1.0 / scale_y, 1.0 / scale_x)
            kernel_size = stride
        else:
            stride, kernel_size, _ = (_triple(module.stride), _triple(module.kernel_size), _triple(module.padding))

        if not torch.is_grad_enabled():  # type:ignore
            # Convert strided convolutions/pooling to average pool
            if (
                isinstance(module, (torch.nn.MaxPool2d))
                or (stride[0] > 1 and stride[0] > kernel_size[0])
                or (stride[1] > 1 and stride[1] > kernel_size[1])
                or (stride[2] > 1 and stride[2] > kernel_size[2])
            ):
                # Pytorch documentation is explicitely against changing output in a forward hook
                # However, since we do not really need the graph or gradients to be correct
                # it shouldn't harm.
                if module.padding != 0:
                    padding = module.padding
                    if not isinstance(module.padding, tuple):
                        padding = [module.padding, module.padding]
                    padded_input = torch.nn.functional.pad(inpt[0], [padding[1], padding[1], padding[0], padding[0]])
                else:
                    padded_input = inpt[0]

                new_output = torch.nn.functional.avg_pool2d(padded_input, kernel_size[1:], stride[1:])
                new_output = torch.sum(new_output, dim=1)[0]
                new_output = new_output > (1 - self.eps) * new_output.max()
                new_output = new_output.expand_as(output[0])

                output[0] = new_output.type(self.dtype)

            # Sum all dimensions (useful for DenseNet like networks)
            lost = self._non_max_border_amount(output)

            # Make output between 0-1 again, so the values do not explode
            output.fill_(0)
            output[
                :, :, lost.top : output[0, 0].shape[0] - lost.bottom, lost.left : output[0, 0].shape[1] - lost.right
            ] = 1

            module_stats = {"lost": lost, "stride": stride, "module": module}
            if self.verbose:
                print(module, "\n", module_stats["lost"])

            self._saved_tensors[module] = inpt
            self._module_stats[module] = module_stats
        else:
            module_stats = self._module_stats[module]

            p_stats = self._prev_stats(output)
            if p_stats:
                output_stride = p_stats["output_stride"] * torch.tensor(p_stats["stride"])
            else:
                output_stride = torch.tensor([1, 1, 1])

            module_stats["output_stride"] = output_stride.clone().detach()
            self._stats_per_grad_fn[output.grad_fn] = module_stats
            self._module_stats[module] = module_stats

    def _backward_gather_statistics_hook(self, module, grad_in, grad_out):
        if isinstance(module, (torch.nn.Upsample, StreamingUpsample)):
            if module.mode != "bilinear":
                raise ValueError("Streaming statistics only support bilinear upsample.")
            if grad_in[0] is not None:
                scale_y = float(grad_out[0].shape[H_DIM]) / float(grad_in[0].shape[H_DIM])
                scale_x = float(grad_out[0].shape[W_DIM]) / float(grad_in[0].shape[W_DIM])
            else:
                if module.scale_factor is None:
                    scale_y, scale_x = 1.0, 1.0
                else:
                    scale_y, scale_x = self._upsample_scale_factors(module, (grad_out[0],), grad_out[0])
            stride = (1.0, 1.0 / scale_y, 1.0 / scale_x)
            kernel_size = stride
        else:
            stride, kernel_size, _ = (_triple(module.stride), _triple(module.kernel_size), _triple(module.padding))
        if grad_in[0] is not None:
            # We sum over the channels to deal with networks that do different operations
            # on groups of channels
            f_grad = torch.sum(grad_in[0], dim=1)[0]
            if isinstance(module, (torch.nn.MaxPool2d)):
                # MaxPool shifts indices around, which break the calculation to
                # find valid gradient values. To fix this we do an average pool
                # with the same kernel-size and stride and repeat using the stride.

                # We have the input gradient for a max pool right now
                # The following computes the input gradient for an average pool instead
                # compute the output of the average pool, and multiply by output gradient

                # from forward statistics hook, modified input
                inpt = self._saved_tensors[module]
                padded_inpt = inpt[0]

                if module.padding != 0:
                    padded_inpt = torch.nn.functional.pad(
                        inpt[0], [module.padding, module.padding, module.padding, module.padding], value=-1
                    )

                new_outpt = torch.nn.functional.avg_pool2d(padded_inpt, kernel_size[1:], stride[1:])[0]
                new_outpt = torch.sum(new_outpt, dim=0)

                f_grad = torch.sum(grad_out[0], dim=1)[0]
                f_grad = f_grad * new_outpt
                f_grad = f_grad.cpu()
                f_grad = np.repeat(f_grad, stride[1], axis=0)
                f_grad = np.repeat(f_grad, stride[2], axis=1)
                grad = np.zeros(grad_in[0].shape[2:])

                print("testing shape gradient fix")
                grad[: f_grad.shape[0], : f_grad.shape[1]] = f_grad[: grad.shape[0], : grad.shape[1]]

                f_grad = torch.from_numpy(grad)
                f_grad = f_grad.to(self.device)

            grad_lost = self._non_max_border_amount(grad_out[0])

            if self.verbose:
                print(module, "\n", grad_lost)
            self._module_stats[module]["grad_lost"] = grad_lost

            valid_grad = f_grad > (1 - self.eps) * f_grad.max()

            # When kernel_size > stride we have some _overlap_ of gradients,
            # this overlap makes extra positions in the input gradient invalid
            if (
                (stride[0] > 1 and kernel_size[0] > stride[0])
                or (stride[1] > 1 and kernel_size[1] > stride[1])
                or (stride[2] > 1 and kernel_size[2] > stride[2])
            ):
                valid_lost = self._non_max_border_amount(f_grad)
                valid_grad.fill_(0)
                overlap_rows = kernel_size[1] - stride[1]
                overlap_cols = kernel_size[2] - stride[2]
                valid_grad[
                    valid_lost.top + overlap_rows : valid_grad.shape[0] - valid_lost.bottom - overlap_rows,
                    valid_lost.left + overlap_cols : valid_grad.shape[1] - valid_lost.right - overlap_cols,
                ] = 1

            new_grad_in = valid_grad[None].expand(grad_in[0].shape[1], *valid_grad.shape)[None]
            new_grad_in = new_grad_in.type(self.dtype) * 10 - 1
            new_grad_in_lost = self._non_max_border_amount(new_grad_in)

            return (new_grad_in, *grad_in[1:])
        return None

    def _backward_saliency_hook(self, module: StreamingConv2d, grad_in, grad_out, is_bias=False, change_grad=True):
        stride: List[int] = _triple(module.stride)  # type:ignore

        # Trim gradient of invalid values
        sides = module.input_loc.sides
        grad_lost = module.grad_lost  # type: Lost

        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0
        lost = Lost(lost_top, lost_left, lost_bottom, lost_right)

        grad = grad_out[0]
        valid_grad = grad[:, :, lost_top : grad.shape[H_DIM] - lost_bottom, lost_left : grad.shape[W_DIM] - lost_right]

        output_stride = module.output_stride * torch.tensor(stride)
        stride_y = self._stride_value(output_stride, 1)
        stride_x = self._stride_value(output_stride, 2)
        input_loc = module.input_loc

        # Move the location according to how many pixels have been trimmed
        # this will be the location of the valid gradient of this layer in relation
        # to the actual gradient in a normal backpass
        data_loc_y = self._floor_div(input_loc.y, stride_y) + lost_top
        data_loc_x = self._floor_div(input_loc.x, stride_x) + lost_left

        data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)

        # Calculate which part of the gradient is 'new'
        old_value_indices = self.saliency_old_indices
        new_output_box, updated_total_indices = _new_value_indices(
            valid_grad.shape, data_loc, old_value_indices
        )

        if module.in_channels == 3:
            valid_grad_in = grad_in[0][
                :,
                :,
                lost.top * stride[1] : grad_in[0].shape[2] - lost.bottom * stride[1],
                lost.left * stride[2] : grad_in[0].shape[3] - lost.right * stride[2],
            ]

            relevant_input_grad = valid_grad_in[
                :,
                :,
                new_output_box.y * stride[1] : new_output_box.y * stride[1] + new_output_box.height * stride[1],
                new_output_box.x * stride[2] : new_output_box.x * stride[2] + new_output_box.width * stride[2],
            ]

            self.saliency_map[
                :,
                :,
                updated_total_indices.y * stride[1] : updated_total_indices.height * stride[1],
                updated_total_indices.x * stride[2]
                - relevant_input_grad.shape[3] : updated_total_indices.x * stride[2],
            ] = relevant_input_grad.detach().cpu()

            del relevant_input_grad
            del valid_grad_in
        return grad_in

    def _prev_stats(self, grad_fn):
        """DAG traversal, finds the first grad_fn that is in self._stats_per_grad_fn

        Finds the first grad_fn that is in self._stats_per_grad_fn, which is needed for output stride calculations

        Parameters
        ----------
        grad_fn: the grad function of the current output tensor

        """
        if hasattr(grad_fn, "grad_fn"):
            grad_fn = grad_fn.grad_fn

        prev_stats = None

        if grad_fn in self._stats_per_grad_fn:
            prev_stats = self._stats_per_grad_fn[grad_fn]
            return prev_stats
        elif hasattr(grad_fn, "next_functions") and len(grad_fn.next_functions) > 0:
            children = [x[0] for x in grad_fn.next_functions]

            for x in children:
                prev_stats = self._prev_stats(x)
                if prev_stats is not None:
                    break
            return prev_stats
        return prev_stats

    def get_tile_cache(self):
        named_stats = {"net_stats": {}}
        for name, module in self.stream_module.named_modules():
            if module in self._module_stats:
                named_stats["net_stats"][name] = self._module_stats[module]
        named_stats["output_stride"] = self.output_stride
        named_stats["tile_output_lost"] = self.tile_output_lost  # type:ignore
        named_stats["tile_gradient_lost"] = self.tile_gradient_lost  # type:ignore
        named_stats["tile_output_shape"] = self._tile_output_shape  # type:ignore
        named_stats["output_structure"] = self._output_structure
        return named_stats

    def load_tile_cache(self, state):
        self.disable()

        self.output_stride = state["output_stride"]
        self.tile_output_lost = state["tile_output_lost"]
        self.tile_gradient_lost = state["tile_gradient_lost"]
        self._tile_output_shape = state["tile_output_shape"]
        self._output_structure = state.get("output_structure", self._output_structure)
        if self._output_structure is None and isinstance(self._tile_output_shape, list):
            self._output_structure = ("sequence", list)

        for name, module in self.stream_module.named_modules():
            if name in state["net_stats"]:
                self._module_stats[module] = state["net_stats"][name]

        self.enable()

    def __call__(self, image, **kwargs):
        result_on_cpu = kwargs.pop("result_on_cpu", False)
        return self.forward(image, result_on_cpu)
