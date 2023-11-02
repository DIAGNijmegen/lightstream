from torchmetrics import Metric
from torch import Tensor
import torch
import pandas as pd
from lifelines import CoxPHFitter
import numpy as np

class HazardRatio(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("event", default=[], dist_reduce_fx="cat")
        self.add_state("output", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.cph_model = CoxPHFitter()

    def update(self, output: Tensor, event: Tensor, time: Tensor):

        self.event.append(event)
        self.output.append(output)
        self.time.append(time)

    def compute(self):

        df = pd.DataFrame({"output": np.array(self.output).flatten(),
                           "event": np.array(self.event).flatten(),
                           "time": np.array(self.time).flatten()})

        self.cph_model.fit(df=df, duration_col="time", event_col="event")
        return torch.as_tensor(self.cph_model.hazard_ratios_["output"])

if __name__ == "__main__":
    label = [0,1,2,3]
    event = [0, 1,3,3]
    time = [6, 13, 25, 37 ]

    metric = HazardRatio()
    metric.update(event, label, time)
    print(metric.compute())
    print(torch.as_tensor(0))