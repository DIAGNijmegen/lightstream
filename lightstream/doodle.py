from torchmetrics import Metric
from torch import Tensor
import torch
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored


class HazardRatio(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("event", default=[], dist_reduce_fx="cat")
        self.add_state("output", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")

    def update(self, output: Tensor, event: Tensor, time: Tensor):
        self.event.append(event)
        self.output.append(output)
        self.time.append(time)

    def compute(self):
        self.event = np.array(self.event).flatten()
        self.output = np.array(self.output).flatten()
        self.time = np.array(self.time).flatten()

        print(self.event, self.output, self.time)
        c_index, concordant, disconcordant, tied_risk, tied_time = concordance_index_censored(
            self.event, self.time, self.output
        )
        return torch.as_tensor(c_index)


if __name__ == "__main__":
    label = [0, 1, 2, 2, 1, 0]
    event = [True, True, True, False, True, True]
    time = [0.1, 0.9, 2.4, 1.9, 1, 0.1]

    metric = HazardRatio()
    metric.update(time, event, label)
    print(metric.compute())
