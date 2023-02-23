# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import json
import os
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
from deepsparse import Engine, Scheduler
from deepsparse.utils import generate_random_inputs, model_to_path, override_onnx_input_shapes
from squad_QSL import get_squad_QSL

MAX_SEQ_LEN = 384

def batched_list(lst, n):
    """Yield successive n-sized chunks from lst, with the last possibly short."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def scenario_to_scheduler(scenario):
    if scenario == "SingleStream":
        return Scheduler.single_stream
    elif scenario == "MultiStream":
        return Scheduler.single_stream
    elif scenario == "Offline":
        return Scheduler.single_stream
    elif scenario == "Server":
        return Scheduler.multi_stream
    else:
        raise Exception(scenario)

class Item:
    def __init__(self, query_id, feature):
        self.query_id = query_id
        self.feature = feature

def create_engines(model_path, batch_size, scheduler, sequence_lengths):
    engines = {}
    for seq_len in sorted(sequence_lengths):
        eng = Engine(model_path, batch_size=batch_size, scheduler=scheduler, input_shapes=[[batch_size, seq_len]])
        engines[seq_len] = eng
    return engines

class BERT_DeepSparse_SUT():
    def __init__(self, args):
        self.profile = args.profile
        self.model_path = model_to_path(args.model_path)
        self.batch_size = args.batch_size
        self.scenario = args.scenario
        self.scheduler = scenario_to_scheduler(args.scenario)
        self.sequence_lengths = [64, 128, 192, 256, MAX_SEQ_LEN]
        # self.sequence_lengths = [MAX_SEQ_LEN]

        print("Loading ONNX model...", self.model_path)
        self.engines = create_engines(self.model_path, batch_size=self.batch_size, scheduler=self.scheduler, sequence_lengths=self.sequence_lengths)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(total_count_override=args.max_examples, unpadding_lengths=self.sequence_lengths)

        print("Warming up engine...")
        with override_onnx_input_shapes(self.model_path, input_shapes=[[self.batch_size, MAX_SEQ_LEN]]) as model_path:
            warmup_inputs = generate_random_inputs(model_path, self.batch_size)
            for i in range(5):
                self.predict(warmup_inputs, MAX_SEQ_LEN)

    def predict(self, input, sequence_length):
        # Choose the right engine and run
        return self.engines[sequence_length].run(input, val_inp=False)

    def pad_to_batch(self, x):
        x_pad = np.pad(x, ((0,self.batch_size-x.shape[0]), (0,0)))
        return x_pad

    def process_batch(self, batched_features):
        pad_func = lambda x: self.pad_to_batch(x) if len(batched_features) != self.batch_size else x
        fd = [
            pad_func(np.stack(
                np.asarray([f.feature.unpadded_input_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :, :]),
            pad_func(np.stack(
                np.asarray([f.feature.unpadded_input_mask for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :, :]),
            pad_func(np.stack(
                np.asarray([f.feature.unpadded_segment_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :, ])
        ]
        return fd

    def issue_queries(self, query_samples):
        if self.scenario == "SingleStream" or self.scenario == "Server":
            for i in range(len(query_samples)):
                eval_features = self.qsl.get_features(query_samples[i].index)
                fd = [eval_features.unpadded_input_ids[np.newaxis, :],
                    eval_features.unpadded_input_mask[np.newaxis, :],
                    eval_features.unpadded_segment_ids[np.newaxis, :]]

                scores = self.predict(fd, eval_features.min_pad_length)

                output = np.stack(scores, axis=-1)[0]
                if output.shape[0] < MAX_SEQ_LEN:
                    output = np.pad(output, ((0,MAX_SEQ_LEN-output.shape[0]), (0,0)))

                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])

        elif self.scenario == "Offline":
            # Extract features from queries and split into buckets
            bucketed_features = {seqlen: [] for seqlen in self.sequence_lengths}
            for i in range(len(query_samples)):
                eval_feature = self.qsl.get_features(query_samples[i].index)
                bucketed_features[eval_feature.min_pad_length].append(Item(query_samples[i].id, eval_feature))

            for bucket_seq_len, bucket_eval_features in bucketed_features.items():
                batch_ind = 0
                for batch_ind, batched_features in enumerate(batched_list(bucket_eval_features, self.batch_size)):
                    unpadded_batch_size = len(batched_features)
                    fd = self.process_batch(batched_features)

                    scores = self.predict(fd, bucket_seq_len)

                    output = np.stack(scores, axis=-1)
                    if output.shape[1] < MAX_SEQ_LEN:
                        output = np.pad(output, ((0,0), (0,MAX_SEQ_LEN-output.shape[1]), (0,0)))

                    # sending responses individually
                    for sample in range(unpadded_batch_size):
                        response_array = array.array("B", output[sample].tobytes())
                        bi = response_array.buffer_info()
                        lg.QuerySamplesComplete([lg.QuerySampleResponse(batched_features[sample].query_id, bi[0], bi[1])])

        else:
            raise Exception("Unknown scenario", scenario)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_deepsparse_sut(args):
    return BERT_DeepSparse_SUT(args)
