# coding:utf-8
import json
import penman
import smatch
import sys

from pathlib import Path
from postprocessing_amr import decode_amr
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Tuple, Union


def calculate_smatch(test_path, predictions_path) -> dict:
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return {"smatch": score[2]}


pred_file = sys.argv[1]
gold_file = sys.argv[2]


with open(pred_file, "r", encoding="utf-8") as fin:
    pred_data = [f'<s> {itm.strip().replace("<|end_of_text|>", "")} </s>' for itm in fin]

graphs = []
for ith_pred in pred_data:
    graphs_same_source = []
    graphs.append(graphs_same_source)
    graph, status, (lin, backr) = decode_amr(
        ith_pred, restore_name_ops=False
    )
    graph.status = status
    graph.nodes = lin
    graph.backreferences = backr
    graph.tokens = ith_pred
    graphs_same_source.append(graph)

graphs_same_source[:] = tuple(
    zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0])))
)[1]

idx = 0
for gps in graphs:
    # print("graph-sent:", gps, snt)
    for gp in gps:
        metadata = {}
        metadata["id"] = str(idx)
        metadata["annotator"] = "bart-amr"
        # metadata["date"] = str(datetime.datetime.now())
        metadata["snt"] = "None"
        if "save-date" in metadata:
            del metadata["save-date"]
        gp.metadata = metadata
        idx += 1

print("Before Penman Encoding")
pieces = [penman.encode(g[0]) for g in graphs]

output_prediction_file = f"{pred_file}.processed"

# write predictions and targets for later rouge evaluation.
with open(output_prediction_file, "w") as p_writer:
    p_writer.write("\n\n".join(pieces))

assert os.path.isfile(gold_file), "Invalid gold file path {gold_file}, file not existed"

try:
    smatch_score = calculate_smatch(gold_file, output_prediction_file)
except:
    smatch_score = {"smatch": 0.0}

print("Smatch_score:", smatch_score)