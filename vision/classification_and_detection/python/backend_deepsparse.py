"""
DeepSparse Inference Engine backend (https://github.com/neuralmagic/deepsparse)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import os
import numpy as np

import deepsparse
from deepsparse.utils import get_input_names, get_output_names, model_to_path

import backend


class BackendDeepsparse(backend.Backend):
    def __init__(self):
        super(BackendDeepsparse, self).__init__()
        self.engine = None
        self.max_batchsize = 1
        self.pool = None

    def version(self):
        return deepsparse.__version__

    def name(self):
        """Name of the runtime."""
        return "deepsparse"

    def image_format(self):
        """image_format. For onnx it is always NCHW."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""

        # If the model is a SparseZoo stub, download it and get new path
        model_path = model_to_path(model_path)

        self.engine = deepsparse.Engine(model=model_path, batch_size=self.max_batchsize)

        self.inputs = inputs if inputs else get_input_names(model_path)
        self.outputs = outputs if outputs else get_output_names(model_path)

        return self

    def predict(self, feed):
        """Run the prediction"""
        engine_inputs = list(feed.values())
        engine_outputs = self.engine.run(engine_inputs)
        return engine_outputs
