"""Model and layer conversion helpers for the streaming engine."""
import logging

import torch

from lightstream.core.scnn.utils import Lost
from lightstream.core.scnn.streamingconv import StreamingConv2d
from lightstream.core.scnn.streamingupsample import StreamingUpsample2d
from lightstream.core.reducer import BaseReducer, BaseStreamingGlobalReducer

logger = logging.getLogger(__name__)

class ConversionMixin:
    def _convert_modules_for_streaming(self, module):
        mod = module
        if isinstance(module, torch.nn.Conv2d):
            if module in self._module_stats:
                mod = StreamingConv2d.from_torch_conv2d(module)
                mod.grad_lost = self._module_stats[module]["grad_lost"]
                mod.output_stride = self._module_stats[module]["output_stride"]
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, torch.nn.Upsample):
            mod = StreamingUpsample2d.from_torch_upsample(module)
            if module in self._module_stats:
                mod.grad_lost = self._module_stats[module].get("grad_lost", Lost(0, 0, 0, 0))
                mod.output_stride = self._module_stats[module].get("output_stride", torch.tensor([1, 1, 1]))
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, BaseReducer):
            mod = module.to_streaming()
            self._streaming_reducers.append(mod)
        for name, child in module.named_children():
            mod.add_module(name, self._convert_modules_for_streaming(child))
        del module
        return mod

    def _reset_converted_modules(self, module):
        mod = module
        if isinstance(module, StreamingConv2d):
            mod = module.to_torch_conv2d()
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, StreamingUpsample2d):
            mod = module.to_torch_upsample()
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, BaseStreamingGlobalReducer):
            mod = module.to_reducer()
        for name, child in module.named_children():
            mod.add_module(name, self._reset_converted_modules(child))
        del module
        return mod
