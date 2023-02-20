# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
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

import os
import sys
sys.path.insert(0, os.getcwd())

from transformers import BertTokenizer
from create_squad_data import read_squad_examples, convert_examples_to_features

import mlperf_loadgen as lg

# To support feature cache.
import pickle

max_seq_length = 384
max_query_length = 64
doc_stride = 128

class SQuAD_v1_QSL():
    def __init__(self, total_count_override=None, perf_count_override=None, cache_path='eval_features.pickle', unpadding_lengths=None):
        print("Constructing QSL...")
        eval_features = []
        # Load features if cached, convert from examples otherwise.
        if os.path.exists(cache_path):
            print("Loading cached features from '%s'..." % cache_path)
            with open(cache_path, 'rb') as cache_file:
                eval_features = pickle.load(cache_file)
        else:
            print("No cached features at '%s'... converting from examples..." % cache_path)

            print("Creating tokenizer...")
            vocab_file = os.environ.get("VOCAB_FILE")
            if not vocab_file:
                vocab_file = "build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
            tokenizer = BertTokenizer(vocab_file)

            print("Reading examples...")
            dataset_file = os.environ.get("DATASET_FILE")
            if not dataset_file:
                dataset_file = "build/data/dev-v1.1.json"
            eval_examples = read_squad_examples(input_file=dataset_file,
                is_training=False, version_2_with_negative=False)

            print("Converting examples to features...")
            def append_feature(feature):
                eval_features.append(feature)

            convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=False,
                output_fn=append_feature,
                verbose_logging=False)

            print("Caching features at '%s'..." % cache_path)
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(eval_features, cache_file)

        # Set up metadata on padding for eval_features
        if unpadding_lengths:
            import numpy as np

            unpadding_lengths = sorted(unpadding_lengths)

            def find_min_length(feature_array):
                for sl in unpadding_lengths:
                    if len(feature_array) <= sl:
                        return sl
                # If we don't find anything, use the largest
                return unpadding_lengths[-1]

            for i in range(len(eval_features)):
                # Trim all the zeros from the back aka padding
                padded_input_ids = np.array(eval_features[i].input_ids).astype(np.int64)
                padded_input_mask = np.array(eval_features[i].input_mask).astype(np.int64)
                padded_segment_ids = np.array(eval_features[i].segment_ids).astype(np.int64)
                unpadded_input_ids = np.trim_zeros(padded_input_ids, 'b')
                unpadded_input_mask = np.trim_zeros(padded_input_mask, 'b')
                unpadded_segment_ids = np.trim_zeros(padded_segment_ids, 'b')

                assert len(unpadded_input_ids) == len(unpadded_input_mask) == len(unpadded_segment_ids)

                min_pad_length = find_min_length(unpadded_input_ids)
                min_padding = min_pad_length - len(unpadded_input_ids)

                eval_features[i].min_pad_length = min_pad_length
                eval_features[i].unpadded_input_ids = np.pad(unpadded_input_ids, (0, min_padding))
                eval_features[i].unpadded_input_mask = np.pad(unpadded_input_mask, (0, min_padding))
                eval_features[i].unpadded_segment_ids = np.pad(unpadded_segment_ids, (0, min_padding))

        self.eval_features = eval_features
        self.count = total_count_override or len(self.eval_features)
        self.perf_count = perf_count_override or self.count
        self.qsl = lg.ConstructQSL(self.count, self.perf_count, self.load_query_samples, self.unload_query_samples)
        print("Finished constructing QSL.")

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass

    def get_features(self, sample_id):
        return self.eval_features[sample_id]

    def __del__(self):
        print("Finished destroying QSL.")

def get_squad_QSL(total_count_override=None, perf_count_override=None, unpadding_lengths=None):
    return SQuAD_v1_QSL(total_count_override, perf_count_override, unpadding_lengths=unpadding_lengths)
