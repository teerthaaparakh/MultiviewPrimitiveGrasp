import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator


class MyDatasetEvaluator(DatasetEvaluator):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def process(self, inputs, outputs):
        # inputs: list of length batch size
        pass

    def evaluate(self):
        pass
