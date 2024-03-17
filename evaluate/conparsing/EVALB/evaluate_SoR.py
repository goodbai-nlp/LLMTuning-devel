import sys
import os
import nltk
import re
from evaluate_bra import *
with open("golds/test.txt","r",encoding="utf-8") as f:
    dataset = [nltk.tree.Tree.fromstring(i.strip()) for i in f.readlines()]
tokens = [i.leaves() for i in dataset]
tags_list = []
for i in dataset:
    tags_list.append(i.pos())
tags = []
for i in tags_list:
    tags.append(j[1] for j in i)
def process_action(action_file):
    with open(action_file,"r",encoding="utf-8") as f:
        datasets = [i.strip() for i in f.readlines()]
    actions = []
    for line in range(len(datasets)):
        if "SHIFT" in datasets[line]:
            datasets[line] = datasets[line][datasets[line].index("SHIFT"):]
    cnt = 0
    for i in datasets:
        token = tokens[cnt]
        tag = tokens[cnt]
        now_cnt = 0
        action = i.split(";")
        temp = []
        for j in action:
            try:
                if "-" in j.strip() and j.strip() != "REDUCE":
                    temp.append("PJ("+j.strip().split("-")[1]+")")
                elif j.strip() == "SHIFT":
                    temp.append(j.strip()+" "+tag[now_cnt]+" "+token[now_cnt])
                    now_cnt += 1
                else:
                    temp.append(j.strip())
            except:
                pass
        actions.append("\n".join(temp))
        cnt += 1
    print("actions length:",len(actions))
    with open(action_file.replace(".txt","_processed.txt"),"w",encoding="utf-8") as f:
        for i in actions:
            f.write(i+"\n\n")

def mid2tree(action_file):
    def tree(acts):
        btree = []
        openidx = []
        wid = 0
        for act in acts:
            if act[0] == 'S':
                tmp = act.split()
                btree.append("("+tmp[1]+" "+tmp[2]+")")
                wid += 1
            elif act[0] == 'P':
                btree.insert(-1,"("+act[3:-1])
                openidx.append(len(btree)-2)
            else:
                tmp = " ".join(btree[openidx[-1]:])+")"
                btree = btree[:openidx[-1]]
                btree.append(tmp)
                openidx = openidx[:-1]
        if len(btree) == 1:
            answers.append(btree[0])
        else:
            answers.append("")
    answers = []
    actions = []
    action = []    
    for line in open(action_file):
        line = line.strip()
        if not line:
            actions.append(action[:-1])
            action = []
        else:
            action.append(line)
    for i in range(len(actions)):
        try:
            tree(actions[i])
        except Exception as e:
            answers.append("")
    print(len(answers))
    with open(action_file.replace(".txt","_tree.txt"),"w",encoding = "utf-8") as f:
        for i in answers:
            f.write(str(i)+"\n")

def evluate(pred_file,gold_file):
    process_action(pred_file)
    mid2tree(pred_file.replace(".txt","_processed.txt"))
    process_pred(pred_file.replace(".txt","_processed_tree.txt"))
    print(pred_file.split(".txt")[0]+"_processed_tree_debug.txt")
    processd_evaluate(pred_file.split(".txt")[0]+"_processed_tree_debug.txt",gold_file)

if __name__ == "__main__":
    pred_file = sys.argv[1]
    gold_file = sys.argv[2]
    evluate(pred_file,gold_file)