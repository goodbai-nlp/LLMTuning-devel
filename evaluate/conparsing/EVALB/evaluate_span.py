from supar.utils.transform import Tree
import nltk
import sys
import re
DELETE = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
DELETE = {'TOP'}
EQUAL = {'ADVP': 'PRT'}
with open(sys.argv[2],"r",encoding="utf-8") as f:
    golds = [nltk.Tree.fromstring(tree) for tree in  f.readlines()]
# preds = [Tree.factorize(tree, DELETE, EQUAL) for tree in preds]
golds_span = []
#trees = [Tree.factorize(tree, DELETE, EQUAL) for tree in golds]
trees = [Tree.factorize(tree,DELETE) for tree in golds]
for i,j in zip(trees,golds):
    q = j.leaves()
    temps = []
    for temp in i:
        temps.append((" ".join(q[temp[0]:temp[1]]),temp[2]))
    golds_span.append(temps)
def extract(text):
    pattern = r'"([^"]+)" is a (\w+)\.'
    matches = re.findall(pattern, text)
    return matches
cnt = 0
preds_span = []
with open(sys.argv[1],"r",encoding="utf-8") as f:
    preds =[i.strip() for i in f.readlines()]
    for i in preds:
        if extract(i):
            preds_span.append(extract(i))
        else:
            preds_span.append([])
gold_cnt = 0
pred_cnt = 0
from collections import Counter
for pred,gold in zip(preds_span,golds_span):
    gold_cnt += len(gold)
    pred_cnt += len(pred)
    pred = Counter(pred)
    gold = Counter(gold)
    cnt += len(list((pred & gold).elements()))
    # for j in pred:
    #     if j not in gold:
    #         print("=======not in gold=========")
    #         print(j)
    # for j in gold:
    #     if j not in pred:
    #         print("=======not in pred=========")
    #         print(j)
recall = cnt/gold_cnt
precision = cnt/pred_cnt
print(gold_cnt)
print(pred_cnt)
print(cnt/gold_cnt)
print(cnt/pred_cnt)
print(2*recall*precision/(recall+precision))
