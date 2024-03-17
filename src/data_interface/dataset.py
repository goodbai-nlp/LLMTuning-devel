# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from dataclasses import dataclass
from datasets import load_dataset
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from torch.utils.data import Subset
import re


def padding_func(
    features,
    padding_side="right",
    pad_token_id=1,
    key="label",
    pad_to_multiple_of=1,
    max_length=None,
):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features) if not max_length else max_length
    if pad_to_multiple_of > 1:
        if max_length is not None:
            max_label_length = min(
                max_length,
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of,
            )
        else:
            max_label_length = (
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    if dataset_name.endswith("webnlg20") or dataset_name.endswith("webnlg17"):
        return WebNLGDatasetVicuna(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("webnlg20new") or dataset_name.endswith("webnlg17new"):
        return WebNLGDatasetVicunaV2(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wsj"):
        return ConParsingDataset(output_path, seed, local_rank, dataset_name, out_format="bracket")
    elif dataset_name.endswith("challenge-qa-llama2") or dataset_name.endswith("_split"):
        return ChallengeQADatasetLlama2(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("webnlgv2.1"):
        return WebNLGV2DatasetLlama2(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("llama2"):
        return InstructionTuningDatasetLlama2(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("amr2text"):
        return AMR2TextDataset(output_path, seed, local_rank, dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "labels" in features[0].keys():
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.label_pad_token_id,
                key="labels",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        return batch

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, data_path, eos_token=""):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # self.eos_token = "<|endoftext|>"
        self.eos_token = eos_token
        self.raw_datasets = load_dataset(data_path)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


class InstructionTuningDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        self.output_path = output_path
        self.dataset_name = "InstructData"
        self.sys_instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "validation": f"{data_path}/valid.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return f'{self.sys_instruction}Human:\n{sample["instruction"]} {sample["input"]}\nAssistant:\n'
    
    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['output']

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample['output']}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": instruction, "input": input})
            for instruction, input in zip(
                samples["instruction"], samples["input"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": instruction, "input": input, "output": output})
            for instruction, input, output in zip(
                samples["instruction"], samples["input"], samples["output"],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class InstructionTuningDatasetVicuna(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        self.output_path = output_path
        self.dataset_name = "InstructData"
        self.sys_instruction = ""
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "validation": f"{data_path}/valid.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        self.indent = " "
        self.input_key = "input"
        self.output_key = "output"
        self.ins_key = "instruction"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return f'{self.sys_instruction}USER:{self.indent}{sample["instruction"]} {sample["input"]}{self.indent}ASSISTANT:'

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['output']

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{self.indent}{sample['output']}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": instruction, "input": input})
            for instruction, input in zip(
                samples[self.ins_key], samples[self.input_key]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": instruction, "input": input, "output": output})
            for instruction, input, output in zip(
                samples[self.ins_key], samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class InstructionTuningDatasetLlama2(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        self.output_path = output_path
        self.dataset_name = "InstructData"
        self.sys_instruction = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        self.raw_datasets = load_dataset(
            data_path
        )
        self.input_key = "input"
        self.output_key = "output"
        self.ins_key = "instruction"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return f"[INST] <<SYS>>\n{self.sys_instruction}\n<</SYS>>\n\n{sample['instruction']}{sample['input']} [/INST]"
    
    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['output']

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)} {sample['output']}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": instruction, "input": input})
            for instruction, input in zip(
                samples[self.ins_key], samples[self.input_key]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": instruction, "input": input, "output": output})
            for instruction, input, output in zip(
                samples[self.ins_key], samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class AMR2TextDataset(InstructionTuningDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "amr2textdataset"
        self.sys_instruction = ""
        self.task_instruction = 'Generate a descriptive text for the given abstract meaning representation graph.'


class ConParsingDatasetVicuna(InstructionTuningDatasetVicuna):
    def __init__(self, output_path, seed, local_rank, data_path, out_format="bracket"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "conparsing"
        self.sys_instruction = ""
        self.task_instruction = "Generate the constituent tree for a given sentence."
        self.input_key = "sentence"
        assert out_format in ["bracket", "SoR"]
        self.output_key = out_format

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": self.task_instruction, "input": input})
            for input in samples[self.input_key]
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": self.task_instruction, "input": input, "output": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class ConParsingDataset(InstructionTuningDataset):
    def __init__(self, output_path, seed, local_rank, data_path, out_format="bracket"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "conparsing"
        self.sys_instruction = ""
        self.task_instruction = "Generate the constituent tree for a given sentence."
        self.input_key = "sentence"
        assert out_format in ["bracket", "SoR"]
        self.output_key = out_format

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": self.task_instruction, "input": input})
            for input in samples[self.input_key]
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": self.task_instruction, "input": input, "output": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}
    

class ChallengeQADatasetLlama2(InstructionTuningDatasetLlama2):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "ChallengeQA"
        self.task_instruction = ""
        self.input_key = "Question"
        self.output_key = "Correct Answer"

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": self.task_instruction, "input": input})
            for input in samples[self.input_key]
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": self.task_instruction, "input": input, "output": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}

webnlg_template="""<s>[INST] Following is a set of knowledge graph triples delimited by triple backticks, each on a separate line, in the format: subject | predicate | object.
```
{triples}
```

Generate a coherent piece of text that contains all of the information in the triples. Only use information from the provided triples.[/INST]"""


class WebNLGV2DatasetLlama2(InstructionTuningDatasetLlama2):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "WebNLGv2.1"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "validation": f"{data_path}/valid.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        self.task_instruction = ""
        self.input_key = "input"
        self.output_key = "output"

    def get_prompt(self, sample):
        return webnlg_template.format(triples=sample["input"])
    
    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample['output']}"
        
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": self.task_instruction, "input": input})
            for input in samples[self.input_key]
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": self.task_instruction, "input": input, "output": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}
    

class WebNLGDatasetVicuna(InstructionTuningDatasetVicuna):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "WebNLG"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "validation": f"{data_path}/valid.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        self.task_instruction = "Generate a coherent piece of text that contains all of the information in the triples."
        self.input_key = "input"
        self.output_key = "output"

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": self.task_instruction, "input": input})
            for input in samples[self.input_key]
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": self.task_instruction, "input": input, "output": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}
    
    
class WebNLGDatasetVicunaV2(InstructionTuningDatasetVicuna):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "WebNLG"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "validation": f"{data_path}/valid.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        # self.task_instruction = "Following is a set of knowledge graph triples delimited by triple backticks, each on a separate line, in the format: subject | predicate | object. Generate a coherent piece of text that contains all of the information in the triples."
        self.task_instruction = "Generate a coherent piece of text that contains all of the information in the triples."
        self.input_key = "input"
        self.output_key = "output"

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"instruction": self.task_instruction, "input": input})
            for input in samples[self.input_key]
        ]
        input_full = [
            self.get_prompt_and_chosen({"instruction": self.task_instruction, "input": input, "output": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key],
            )
        ]
        return {"text": input_full, "prompt": input_prompt}