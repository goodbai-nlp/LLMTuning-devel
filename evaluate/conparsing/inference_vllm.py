# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import deepspeed
import logging
import torch
import json
import sys
import os
from nltk import Tree
from tqdm import tqdm
from transformers import (
    AutoConfig,
    LlamaConfig,
    LlamaTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
)
from vllm import LLM, SamplingParams
basepath=os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(basepath)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test_file",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        default=False,
        help="use beam search or not",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", default=False, help="whether to use deepspeed for inference"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        help="testset",
        required=True,
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Generate the constituent tree for a given sentence.",
        help="testset",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="None",
        help="testset",
    )
    parser.add_argument(
        "--bit_8",
        action="store_true", default=False
    )

    args = parser.parse_args()

    return args


def generate(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=False,
    num_return_sequences=1,
    max_new_tokens=100,
):
    generate_ids = model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )
    try:
        result = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except IndexError:
        result = []
        print("Invaid index in generated ids:", generate_ids)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def post_process(gen_res):
    res = []
    for prompt, pred in gen_res:
        cleaned = pred.replace(prompt, "").replace("\n", "").split("<|endoftext|>")[0]
        res.append(cleaned)
    return res


def prompt_eval(args, model, tokenizer, device, prompts):
    all_res = []
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_res = generate(
            model,
            tokenizer,
            inputs,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
        )
        if len(gen_res):
            all_res.append((prompt, gen_res[0]))
        else:
            all_res.append((prompt, prompt + "(Error in Str)"))
        # print(gen_res)
    pred_cleaned = post_process(all_res)
    return pred_cleaned


def tree_to_lisp(tree):
    """
    Converts an NLTK Tree to a Lisp-style string.
    """
    if isinstance(tree, Tree):
        children_str = ' '.join(tree_to_lisp(t) for t in tree)
        return f"({tree.label()} {children_str})"
    else:
        return tree


world_size = torch.cuda.device_count()


def main():
    args = parse_args()
    model = LLM(model=args.model_name if args.model_name else args.model_name_or_path, download_dir=args.model_name_or_path, tensor_parallel_size=world_size, gpu_memory_utilization=0.80)
    print(args.test_file)
    with open(args.test_file, "r", encoding="utf-8") as fin:
        if args.prompt_template == "None":
            prompts = [json.loads(line.strip())["sentence"] for line in fin]
        elif args.prompt_template == "supervised":
            prompts = [
                f" Human: {args.instruction} {json.loads(line.strip())['sentence']} Assistant: "
                for line in fin
            ]
        elif args.prompt_template == "supervised-conparsing":
            prompts = [
                f"Human:\n{args.instruction} {json.loads(line.strip())['sentence']}\nAssistant:\n"
                for line in fin
            ]
        elif args.prompt_template == "supervised-conparsing-new":
            sys_instruction=""
            task_instruction="Generate the constituent tree for a given sentence."
            prompts = [
                    f"{sys_instruction}USER: {task_instruction} {json.loads(line.strip())['sentence']} ASSISTANT:"
                for line in fin
            ]
        elif args.prompt_template == "supervised-conparsing-code":
            prompts = [
                f'"""\n{args.instruction}\n{json.loads(line.strip())["sentence"]}\n"""\n(TOP (S ('
                for line in fin
            ]
        elif args.prompt_template == "supervised-conparsing-pycode":
            sys_prompt = "".join(open("./pycode_prompt.txt", 'r', encoding='utf-8').readlines()).strip()
            args.instruction = "Represent the constituent parse tree of the following sentence into a nested parentheses format."
            prompts = [
                f'{sys_prompt}\n\n"""\n{args.instruction}\n{json.loads(line.strip())["input"]}\n"""\ntree_instance='
                for line in fin
            ]
        elif args.prompt_template == "vicuna":
            prompts = [
                f"USER: {args.instruction}{json.loads(line.strip())['sentence']} ASSISTANT:"
                for line in fin
            ]
        elif args.prompt_template == "gpt":
            prompts = [
                f"Human:\n{args.instruction}{json.loads(line.strip())['sentence']}\nAssistant:\n"
                for line in fin
            ]
        elif args.prompt_template == "one_shot":
            prompts = [
                f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: {json.loads(line.strip())['sentence']}\n### Assistant:"
                for line in fin
            ]
        elif args.prompt_template == "one_shot_sim":
            prompts = [
                f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: {json.loads(line.strip())['sentence']}\n### Assistant:"
                for line in fin
            ]
        elif args.prompt_template == "ccgbank":
            prompts = [
                f" Human: {args.instruction} {json.loads(line.strip())['src']} Assistant: "
                for line in fin
            ]
        else:
            print("Invalid Prompt template, exit ...")
            exit()
    
    print(f"Example data: {prompts[:5]}")
       
    if args.beam_search:
        gen_params = SamplingParams(n=args.num_beams, use_beam_search=True, temperature=0.0, best_of=args.num_beams, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>"])
    else:
        #gen_params = SamplingParams(n=1, use_beam_search=False, best_of=1, temperature=0, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>"])
        #gen_params = SamplingParams(n=1, use_beam_search=False, best_of=1, temperature=0, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>", "###"])
        gen_params = SamplingParams(n=1, use_beam_search=False, best_of=1, temperature=0, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>", "###"])
    
    gen_res = ["" for _ in range(len(prompts))]
    prompt_res = ["" for _ in range(len(prompts))]
    outputs = model.generate(prompts, gen_params)
    # print("Output:", outputs)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        if args.prompt_template == "supervised-conparsing-code":
            generated_text = "(TOP (S (" + generated_text
        elif args.prompt_template == "supervised-conparsing-pycode":
            # generated_text = 'Tree(' + generated_text.replace('"', '\"')
            generated_text = generated_text.replace('"', '\"')
            # print("Before exec:", generated_text)
            # tree_instance = eval(generated_text)
            # generated_text = tree_to_lisp(tree_instance)

        iid = int(output.request_id)
        # print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        # print(f"Final Generated text: {generated_text!r}")
        # gen_res.append(generated_text)
        gen_res[iid] = generated_text.replace("\n", "")
        prompt_res[iid] = prompt
    
    out_prefix = args.test_file.split("/")[-1]
    with open(f"{args.model_name_or_path}/{args.out_prefix}_{out_prefix}_pred_fast", "w", encoding="utf-8") as fout:
        fout.write("\n".join(gen_res) + "\n")


if __name__ == "__main__":
    main()
