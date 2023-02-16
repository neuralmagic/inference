"""
DeepSparse Inference Engine backend (https://github.com/neuralmagic/deepsparse)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import numpy as np

import deepsparse
from deepsparse.utils import get_input_names, get_output_names, model_to_path

import backend


def scenario_to_scheduler(scenario):
    if scenario == "SingleStream":
        return deepsparse.Scheduler.single_stream
    elif scenario == "MultiStream":
        return deepsparse.Scheduler.single_stream
    elif scenario == "Offline":
        return deepsparse.Scheduler.multi_stream
    elif scenario == "Server":
        return deepsparse.Scheduler.multi_stream
    else:
        raise Exception(scenario)


class BackendDeepsparse(backend.Backend):
    def __init__(self):
        super(BackendDeepsparse, self).__init__()
        self.engine = None
        # vv Needs to be set by driver script
        self.max_batchsize = None
        self.num_streams = None
        self.scenario = None

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

        scheduler = scenario_to_scheduler(self.scenario)

        self.engine = deepsparse.Engine(
            model=model_path,
            batch_size=self.max_batchsize,
            scheduler=scheduler,
            num_streams=self.num_streams,
        )

        self.inputs = inputs if inputs else get_input_names(model_path)
        self.outputs = outputs if outputs else get_output_names(model_path)

        return self

    def predict_impl(self, feed):

        # Prepare and possibly pad inputs to fit batch
        engine_inputs = []
        max_batch_size = self.max_batchsize
        batch_size = max_batch_size
        for _, data in feed.items():
            batch_size = len(data)
            if batch_size < max_batch_size:
                # Fill in with the first tensor
                data_extra = np.stack([data[0]] * (max_batch_size - batch_size))
                data = np.vstack((data, data_extra))
            elif batch_size > max_batch_size:
                raise ValueError(
                    "Internal MLPerf error: dynamic batch size > max batch size"
                )

            engine_inputs.append(data)

        # Run inference
        engine_outputs = self.engine.run(engine_inputs)

        # Process outputs
        if batch_size < max_batch_size:
            # Take only the output of batch size for dynamic batches
            final_outputs = [out[:batch_size] for out in engine_outputs]
            return final_outputs
        else:
            return engine_outputs

    def predict(self, feed):
        """Run the prediction"""
        return self.predict_impl(feed)
