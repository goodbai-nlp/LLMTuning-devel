from nltk.tree import Tree
from extract_bracket import *
import re
from supar.utils.transform import Tree as suTree
DELETE = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
EQUAL = {'ADVP': 'PRT'}
def error_evaluation(process_list,gold_file,pred_file):
    with open(gold_file,"r",encoding = "utf-8") as f:
        golds = [i.strip() for i in f.readlines()]
    with open(pred_file,"r",encoding = "utf-8") as f:
        preds = [i.strip() for i in f.readlines()]
    cnt = 0
    gold_bracket = 0
    pred_bracket = 0
    for i in process_list:
        gold = suTree.factorize(Tree.fromstring(golds[i-1]), DELETE, EQUAL)
        #gold_word = [i for i in Tree.fromstring(golds[i-1]).leaves() if i not in DELETE]
        gold_word = [i[0] for i in nltk.Tree.fromstring(golds[i-1]).pos() if i[1] not in  DELETE]
        gold = [(" ".join(gold_word[j[0]:j[1]]),j[2]) for j in gold]
        gold_bracket += len(gold)
        try:
            pred = suTree.factorize(Tree.fromstring(preds[i-1]), DELETE, EQUAL)
            #pred_word = [i for i in Tree.fromstring(preds[i-1]).leaves() if i not in DELETE]
            pred_word = [i[0] for i in nltk.Tree.fromstring(preds[i-1]).pos() if i[1] not in  DELETE]
            pred = [(" ".join(pred_word[j[0]:j[1]]),j[2]) for j in pred]
            if len(pred) - len(gold) >= 7:
                node = Tree.fromstring(golds[i-1]).leaves()[-1]
                if node in preds[i-1]:
                    output = preds[i-1][:preds[i-1].index(node)+len(node)]
                    temp = output+(output.count("(")-output.count(")"))*")"
                    pred = suTree.factorize(Tree.fromstring(temp), DELETE, EQUAL)
                    #pred_word = [i for i in Tree.fromstring(preds[i-1]).leaves() if i not in DELETE]
                    pred_word = [j for j in Tree.fromstring(preds[i-1]).leaves()]
                    pred = [(" ".join(pred_word[j[0]:j[1]]),j[2]) for j in pred]
                    pred_bracket += len(pred)
            else:
                pred_bracket += len(pred)
        except Exception as e:
            pred_bracket += 0
        for i in pred:
            if i in gold:
                cnt += 1
    return cnt,gold_bracket,pred_bracket
