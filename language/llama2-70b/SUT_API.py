import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from transformers import LlamaTokenizer

import json
import pickle
import time
import threading
import queue
from concurrent.futures.thread import ThreadPoolExecutor
import more_itertools as mit
from itertools import repeat

import logging
from pathlib import Path

import requests

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("LLM-API-SERVER-SUT")

def except_hook(args):
    print(f"Thread failed with error:")
    print(args.exc_value)
    print(args.exc_traceback)
    os._exit(1)


threading.excepthook = except_hook

gen_kwargs = {
    "max_tokens": 1024,
    "temperature": 0,
}


class SUT():
    def __init__(self,
                 model_path=None,
                 api_server=None,
                 api_model_name=None,
                 dtype="bfloat16",
                 device="cpu",
                 batch_size=None,
                 total_sample_count=24576,
                 dataset_path=None,
                 use_cached_outputs=False,  # Set this to True *only for test accuracy runs* in case your prior session was killed partway through
                 workers=10):

        self.model_path = model_path or api_model_name or "meta-llama/Llama-2-70b-chat-hf"
        self.device = device
        self.api_servers = []
        if api_server:
            self.api_servers.append(api_server)
        self.api_model_name = api_model_name
        self.device = device

        batch_size = total_sample_count
        self.batch_size = batch_size

        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        if 'cuda' in self.device:
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path,
                                   dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count,
                                   device=self.device)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        print(f"LOADING LLAMA TOKENIZER FROM PATH: {self.model_path}")
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            add_prefix_space=False,
            add_bos_token=False,
            use_fast=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_input_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()


    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()


    def query_api_vllm(self, inputs, idx):
        headers = {
            'Content-Type': 'application/json',
        }
        json_data = {
            "model": self.api_model_name,
            "prompt": inputs,
            **gen_kwargs,
        }

        response_code = 0
        print(f"Server path {self.api_servers[idx]}/v1/completions")
        while response_code != 200:
            try:
                response = requests.post(f"{self.api_servers[idx]}/v1/completions", headers=headers, json=json_data, verify=False)
                response_code = response.status_code
            except Exception as e:
                print(e)
                print("connection failure")
                break
        return [resp["text"] for resp in json.loads(response.text)["choices"]]

    def api_action_handler(self, chunk, server_idx):
        output = self.query_api_vllm(chunk, server_idx)
        return output

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """

        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname)
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = None
                tik2 = None
                tik3 = None
                tok = None
            else:
                # Construct / collate batch
                max_seq_len = 1024

                tik1 = time.time()

                input_ids_tensor = []
                input_len = []
                for q in qitem:
                    input_ids_tensor.append(pad(self.data_object.input_ids[q.index],
                                                (max_seq_len - self.data_object.input_lens[q.index], 0, 0, 0),
                                                value=self.tokenizer.pad_token_id))
                    input_len.append(self.data_object.input_lens[q.index])
                input_ids_tensor = torch.cat(input_ids_tensor)

                assert input_ids_tensor.shape[0] <= self.batch_size

                decoded = self.tokenizer.batch_decode(input_ids_tensor)
                cleaned = [entry.replace('</s>','').replace('<s>','') for entry in decoded]
                cleaned_chunks = [list(c) for c in mit.divide(len(self.api_servers), cleaned)]
                
                tik2 = time.time()

                if self.api_servers:
                    with ThreadPoolExecutor(max_workers=len(self.api_servers)) as executor:
                        #needs to be tested
                        output_chunks = list(executor.map(self.api_action_handler,cleaned_chunks,range(len(self.api_servers))))
                    output = []
                    for row in output_chunks:
                        output += row
                else:
                    print("Error: Specify at least one API to which the request is to be sent!")
                    exit(1)

                tik3 = time.time()

                processed_output = np.array(self.tokenizer(output, padding='longest')['input_ids'])

            for i in range(len(processed_output)):
                unpadded = np.delete(processed_output[i], np.where(processed_output[i] == 2))
                n_tokens = unpadded.shape[0]
                response_array = array.array("B", np.array(unpadded, np.int32).tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")
                if tik1:
                    print(f"\tBatchMaker time: {tik2 - tik1}")
                    print(f"\tInference time: {tik3 - tik2}")
                    print(f"\tPostprocess time: {tok - tik3}")
                    print(f"\t==== Total time: {tok - tik1}")
                else:
                    print(f"\tLoaded from cache: {_p}")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl


    def predict(self,**kwargs):
        raise NotImplementedError


    def issue_queries(self, query_samples):
        """ Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        print(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        print(f"IssueQuery done")


    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.first_token_queue = queue.Queue()

    def start(self):

        print(f"Starting {self.num_workers} workers")
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        # Create first token response thread
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.start()


    def process_first_tokens(self):

        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                log.info("Exiting First token response thread")
                break

            first_tokens, response_id = first_token_item

            response_data = array.array("B", np.array(first_tokens, np.float32).tobytes())
            bi = response_data.buffer_info()
            response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
            lg.FirstTokenComplete(response)

    def stream_api_vllm(self, input, response_ids, idx):
        headers = {
            'Content-Type': 'application/json',
        }

        json_data = {
            "model": self.api_model_name,
            "prompt": input,
            "stream": True,
            **gen_kwargs,
            'logprobs': 1
        }

        while True:
            try:
                token_s_cache = []
                s = requests.Session()
                first = True
                with s.post(
                    f'{self.api_servers[idx]}/v1/completions',
                    headers=headers,
                    json=json_data,
                    verify=False,
                    stream=True
                ) as resp:
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode()
                            if decoded.startswith("data") and "[DONE]" not in decoded:
                                data = json.loads(decoded[6:])
                                finish_reason = data["choices"][0]["finish_reason"]
                                stop_reason = data["choices"][0]["stop_reason"]
                                if (finish_reason is not None) or (stop_reason is not None):
                                    if finish_reason == "stop":
                                        token_s = self.tokenizer.eos_token
                                        token_s_cache.append(token_s)
                                    else:
                                        print(f"Sequence finished without hitting eos token, finish_reason: {finish_reason}, stop_reason: {stop_reason}")
                                    continue

                                inter = data["choices"][0]["logprobs"]
                                if "top_logprobs" in inter:
                                    token_s = list(inter["top_logprobs"][0].keys())[0]
                                    if token_s == "":
                                        #print(f"Warning: empty token. Last non-empty token was: \"{token_s_cache[-1]}\"")
                                        continue

                                    if first:
                                        token_ids = self.tokenizer.encode(token_s)
                                        self.first_token_queue.put((token_ids[0], response_ids[0]))
                                        first = False
                                    token_s_cache.append(str(token_s))
                s.close()
                if token_s_cache:
                    # print("Request completed!")
                    #print(token_s_cache)
                    #print("".join(token_s_cache))
                    return self.tokenizer.encode("".join(token_s_cache))
            except Exception as e:
                s.close()
                print(f"Connection failure: {e}")
   
    def async_process_query(self, input_ids_tensor, qitem_id, idx):
        decoded = self.tokenizer.decode(input_ids_tensor[0])
        response_ids = [qitem_id]
        output_tokens = self.stream_api_vllm(decoded, response_ids, idx)

        n_tokens = len(output_tokens)
        if n_tokens <= 1:
            print("WARNING: caught low token count")
            print(input_ids_tensor)
            print(output_tokens)
        response_array = array.array("B", np.array(output_tokens, np.int32).tobytes())
        bi = response_array.buffer_info()
        response = [lg.QuerySampleResponse(qitem_id, bi[0], bi[1], n_tokens)]
        lg.QuerySamplesComplete(response)

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """
        server_idx = 0
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = self.data_object.input_ids[qitem.index]

            if self.api_servers:
                threading.Thread(target=self.async_process_query, args=(input_ids_tensor, qitem.id, server_idx)).start()
            else:
                print("Error: Specify at least one API to which the request is to be sent!")
                exit(1)

    def issue_queries(self, query_samples):
        assert len(query_samples) == 1
        self.query_queue.put(query_samples[0])


    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()
