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

def tree_to_lisp(tree):
    """
    Converts an NLTK Tree to a Lisp-style string.
    """
    if isinstance(tree, Tree):
        children_str = ' '.join(tree_to_lisp(t) for t in tree)
        return f"({tree.label()} {children_str})"
    else:
        return tree
    
with open(sys.argv[1], 'r', encoding="utf-8") as fin:
    res = []
    for idx, line in enumerate(tqdm(fin)):
        try:
            tree_instance = eval(line.strip())
            lisp_string = tree_to_lisp(tree_instance)
        except SyntaxError:
            print(f"line #{idx} has error!")
            lisp_string = ""
        res.append(lisp_string)

with open(sys.argv[2], 'w', encoding="utf-8") as fout:
    fout.write("\n".join(res)+"\n")