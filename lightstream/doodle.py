from torch.utils.data.sampler import WeightedRandomSampler


print(list(WeightedRandomSampler([0.1, 0.5, 0.1, 0.1, 0.1, 0.1], 50, replacement=True)))