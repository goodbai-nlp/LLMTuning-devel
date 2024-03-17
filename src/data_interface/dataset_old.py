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
    max_label_length = max(len(feature[key]) for feature in features)
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
    

@dataclass
class JointDataCollatorForCausalLM:
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
        if "input_view2_ids" in features[0].keys():
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.tokenizer.pad_token_id,
                key="input_view2_ids",
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
        
        if "view2_labels" not in batch:
            batch["view2_labels"] = batch["input_view2_ids"].clone()
        return batch


def get_raw_dataset(dataset_name, output_path, seed, local_rank, out_format="bracket"):
    if dataset_name.endswith("stanford-alpaca"):
        return StanfordAlpacaDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Dahoas/rm-static"):
        return DahoasRmstaticDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Dahoas/full-hh-rlhf"):
        return DahoasFullhhrlhfDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Dahoas/synthetic-instruct-gptj-pairwise"):
        return DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif dataset_name.endswith("yitingxie/rlhf-reward-datasets"):
        return YitingxieRlhfrewarddatasetsDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("openai/webgpt_comparisons"):
        return OpenaiWebgptcomparisonsDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("stanfordnlp/SHP"):
        return StanfordnlpSHPDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wangrui6/Zhihu-KOL"):
        return Wangrui6ZhihuKOLDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Cohere/miracl-zh-queries-22-12"):
        return CohereMiraclzhqueries2212Dataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Hello-SimpleAI/HC3-Chinese"):
        return HelloSimpleAIHC3ChineseDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("mkqa-Chinese"):
        return MkqaChineseDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("mkqa-Japanese"):
        return MkqaJapaneseDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Cohere/miracl-ja-queries-22-12"):
        return CohereMiracljaqueries2212Dataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("lmqg/qg_jaquad"):
        return LmqgQgjaquadDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("lmqg/qag_jaquad"):
        return LmqgQagjaquadDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("webnlg"):
        return WebnlgDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wsj"):
        return ConParsingDataset(output_path, seed, local_rank, dataset_name, out_format)
    elif dataset_name.endswith("wsj-code"):
        return ConParsingCodeDataset(output_path, seed, local_rank, dataset_name, out_format)
    elif dataset_name.endswith("wsj-pycode"):
        return ConParsingPyCodeDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wsj-transition"):
        return ConParsingTransitionDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wsj-joint"):
        return NERConParsingDataset(output_path, seed, local_rank, dataset_name, out_format)
    elif dataset_name.endswith("domain-data"):
        return RawTextDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("pubmed-abs"):
        return PubMedDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("trucated-pubmedqa"):
        return TrucatedPubMedQADataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("pubmedqa"):
        return PubMedQADataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("-amr2text"):
        return AMR2TextDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("-var") or dataset_name.endswith("-leo"):
        return AMRDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("-mtl") or dataset_name.endswith("-mtl-ori"):
        return AMRMTLDataset(output_path, seed, local_rank, dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


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

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return f'{self.sys_instruction}### Human: {sample["instruction"]} {sample["input"]}\n### Assistant: '

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

class InstructionTuningDatasetV2(PromptRawDataset):
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

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
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

class PubMedQADataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "pubmedqa"
        self.dataset_name_clean = "pubmedqa"
        self.input_key = "input"
        self.output_key = "final_decision"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        assert len(sample["CONTEXTS"]) == len(sample["LABELS"])
        context = " ".join([f'{topic.lower()}: {con}' for con, topic in zip(sample["CONTEXTS"], sample["LABELS"])][:250])
        return f"CONTEXT:\n{context}\nQUESTION:\n{sample['QUESTION']}\nANSWER:\n{sample['LONG_ANSWER']}:\nGiven the CONTEXT, QUESTION and ANSWER, judge whether the provided ANSWER correctly addresses the QUESTION under the given CONTEXT. Please output yes, no or maybe. The output is:"

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample[self.output_key]}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"CONTEXTS": context, "QUESTION": question, "LABELS": label, "LONG_ANSWER": answer})
            for context, question, label, answer in zip(
                samples["CONTEXTS"], samples["QUESTION"], samples["LABELS"], samples["LONG_ANSWER"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"CONTEXTS": context, "QUESTION": question, "LABELS": label, "LONG_ANSWER": answer, "final_decision": decision})
            for context, question, label, answer, decision in zip(
                samples["CONTEXTS"], samples["QUESTION"], samples["LABELS"], samples["LONG_ANSWER"], samples["final_decision"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class TrucatedPubMedQADataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "trucated-pubmedqa"
        self.dataset_name_clean = "trucated-pubmedqa"
        self.input_key = "context"
        self.output_key = "final_decision"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return f"{sample['context'].rstrip()}\nQuestion:\n{sample['QUESTION']}\nPlease respond with yes, no or maybe. The answer to the question is:"

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample[self.output_key]}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"context": context, "QUESTION": question})
            for context, question in zip(
                samples["context"], samples["QUESTION"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"context": context, "QUESTION": question, "final_decision": decision})
            for context, question, decision in zip(
                samples["context"], samples["QUESTION"], samples["final_decision"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class PubMedDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "pubmed-abs"
        self.dataset_name_clean = "pubmed-abs"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["abstract"]

    def get_chosen(self, sample):
        return sample["abstract"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include chosen response.")

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = samples["abstract"]
        return {"text": input_text}
    

class RawTextDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "rawtext"
        self.dataset_name_clean = "rawtext"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["text"]

    def get_chosen(self, sample):
        return sample["text"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include chosen response.")

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        return {"text": samples["text"]}


class WebnlgDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "webnlg"
        self.dataset_name_clean = "webnlg"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "validation": f"{data_path}/valid.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        self.instruction = "Generate a descriptive text for the given knowledge graph."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return f"Human:\n{self.instruction} {sample['src']}\nAssistant:\n"

    def get_chosen(self, sample):
        return sample["tgt"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample['tgt']}"
    
    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"src": input, "tgt": output})
            for input, output in zip(
                samples["src"], samples["tgt"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"src": input, "tgt": output})
            for input, output in zip(
                samples["src"], samples["tgt"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class AMRDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, task="amr2text"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "amrdataset"
        self.dataset_name_clean = "amrdataset"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "valid": f"{data_path}/valid.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        self.task = task
        assert self.task in ("amrparsing", "amr2text")
        if self.task == "amr2text":
            self.instruction = "Generate a descriptive text for the given abstract meaning representation graph."
        else:
            self.instruction = "Generate the AMR graph for the given input text."
            
    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["valid"]

    def get_prompt(self, sample):
        if self.task == "amr2text":
            return f"Human:\n{self.instruction} {sample['amr']}\nAssistant:\n"
        else:
            return f"Human:\n{self.instruction} {sample['sentence']}\nAssistant:\n"

    def get_chosen(self, sample):
        return sample["sentence"] if self.task == "amr2text" else sample["amr"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        if self.task == "amr2text":
            return f"{self.get_prompt(sample)}{sample['sentence']}"
        else:
            return f"{self.get_prompt(sample)}{sample['amr']}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"amr": input, "sentence": output})
            for input, output in zip(
                samples["amr"], samples["sentence"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"amr": input, "sentence": output})
            for input, output in zip(
                samples["amr"], samples["sentence"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class AMRMTLDataset(InstructionTuningDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "amrmtldataset"
        self.instruction = 'The following is a linearized directed sematic graph where the nodes such as "z2 important-01" denote concepts and the edges such as ":ARG0-of" denote semantic relations.'


class AMR2TextDataset(InstructionTuningDatasetV2):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "amr2textdataset"
        # self.sys_instruction = 'The following is a linearized directed sematic graph where the nodes such as "z2 important-01" denote concepts and the edges such as ":ARG0-of" denote semantic relations.'
        self.sys_instruction = ""
        self.task_instruction = 'Generate a descriptive text for the given abstract meaning representation graph.'


class ConParsingDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, out_format="bracket"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj"
        self.dataset_name_clean = "wsj"
        self.output_key = out_format
        self.input_key = "sentence"
        assert self.output_key in ["bracket", "SoR"]
        self.instruction = "Generate the constituent tree for a given sentence."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return f"Human:\n{self.instruction} {sample['sentence']}\nAssistant:\n"
    
    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return (
            # f"Human:\n{self.instruction} {sample['sentence']}\nAssistant:\n{sample[self.output_key]}"
        # )
        return (
            f"Human:\n{self.instruction} {sample['sentence']}\nAssistant:\n{sample[self.output_key]}"
        )

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"sentence": input, f"{self.output_key}": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"sentence": input, f"{self.output_key}": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}
    

class ConParsingCodeDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, out_format="bracket"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj"
        self.dataset_name_clean = "wsj"
        self.output_key = out_format
        self.input_key = "sentence"
        assert self.output_key in ["bracket"]
        self.instruction = "Represent the constituent parse tree of the following sentence into a nested parentheses format."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return f'"""\n{self.instruction}\n{sample["sentence"]}\n"""\n(TOP (S ('
    
    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f'"""\n{self.instruction}\n{sample["sentence"]}\n"""\n{sample[self.output_key]}'

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"sentence": input, f"{self.output_key}": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"sentence": input, f"{self.output_key}": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class ConParsingPyCodeDataset(InstructionTuningDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj-pycode"
        self.sys_prompt = "".join(open(f"{data_path}/pycode_prompt.txt", 'r', encoding='utf-8').readlines()).strip()
        self.instruction = "Represent the constituent parse tree of the following sentence as a instance of Tree."

    def get_prompt(self, sample):
        return f'{self.sys_prompt}\n\n"""\n{self.instruction}\n{sample["input"]}\n"""\ntree_instance='
    
    def get_prompt_and_chosen(self, sample):
        return f'{self.sys_prompt}\n\n"""\n{self.instruction}\n{sample["input"]}\n"""\ntree_instance={sample["output"]}'
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"input": input, "output": output})
            for input, output in zip(
                samples["input"], samples["output"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"input": input, "output": output})
            for input, output in zip(
                samples["input"], samples["output"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class ConParsingTransitionDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj"
        self.dataset_name_clean = "wsj"
        self.output_key = "SoR"
        self.input_key = "sentence"
        self.instruction = "Generate the constituent tree for a given sentence."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f" Human: {self.instruction} {sample['sentence']} Assistant: " 

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f" Human: {self.instruction} {sample['sentence']} {sample['pos']} Assistant: {sample[self.output_key]}"
        )

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"sentence": input, "SoR": output, "pos": pos})
            for input, output, pos in zip(
                samples[self.input_key], samples[self.output_key], samples["pos"]
            )
        ]
        return {"text": input_text}


class NERConParsingDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, out_format="bracket"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj"
        self.dataset_name_clean = "wsj"
        self.output_key = out_format
        self.input_key = "sentence"
        assert self.output_key in ["bracket", "SoR"]
        self.instruction = "Generate the constituent tree for a given sentence."
        self.ner_instruction = "Mark all named entities in the given sentence."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f" Human: {self.instruction} {sample['sentence']} Assistant: "
    
    def get_ner_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f" Human: {self.ner_instruction} {sample['sentence']} Assistant: " 

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f" Human: {self.instruction} {sample['sentence']} Assistant: {sample[self.output_key]}"
        )
        
    def get_ner_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f" Human: {self.ner_instruction} {sample['sentence']} Assistant: {sample['ner']}"
        )

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"sentence": input, f"{self.output_key}": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key]
            )
        ]
        input_ner_text = [
            self.get_ner_prompt_and_chosen({"sentence": input, "ner": output})
            for input, output in zip(
                samples[self.input_key], samples['ner']
            )
        ]
        return {
            "text": input_text,
            "ner_text": input_ner_text,
        }


class CAMRDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, eos_token=""):
        super().__init__(output_path, seed, local_rank, data_path, eos_token)
        self.dataset_name = "camr"
        self.dataset_name_clean = "camr"
        self.input_key = "sent"
        self.output_key = "amr"
        self.instruction = "给定如下输入句子以及分词结果，输出句子所对应的中文抽象语义表示图："

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f"{self.instruction} {sample[self.input_key]}\n{sample['context']}<EOU>"

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f"{self.instruction}{sample[self.input_key]}\n{sample['context']}<EOU>{sample[self.output_key]}{self.eos_token}"
        )

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({self.input_key: input, "context": context, f"{self.output_key}": output})
            for input, context, output in zip(
                samples[self.input_key], samples["context"], samples[self.output_key]
            )
        ]
        input_prompt = [
            self.get_prompt({self.input_key: input, "context": context, f"{self.output_key}": output})
            for input, context, output in zip(
                samples[self.input_key], samples["context"], samples[self.output_key]
            )
        ]
        return {"text": input_text, "prompt": input_prompt}
    
