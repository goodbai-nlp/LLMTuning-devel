# coding:utf-8
import json
import sys
from nltk import Tree
from tqdm import tqdm

def parse_lisp_string(lisp_string):
    """
    Parses a Lisp-style string and returns an NLTK Tree.
    """
    tokens = lisp_string.replace('(', ' ( ').replace(')', ' ) ').split()
    res = []
    def parse(tokens):
        if len(tokens) == 0:
            raise ValueError('Unexpected EOF while reading')
        token = tokens.pop(0)
        if token == '(':
            children = []
            while tokens[0] != ')':
                children.append(parse(tokens))
            tokens.pop(0)                   # pop off ')'
            res.append(f"Tree({children[0]}, {children[1:]})")
            return Tree(children[0], children[1:])
        else:
            return token
    tmp = parse(tokens)
    return res[-1]
# res = []
# Example Lisp-style string
lisp_string = "(S (NP (PRP I)) (VP (VBD saw) (NP (DT a) (NN fox))))"
lisp_string = "(TOP (S (NP (DT The) (NN luxury) (NN auto) (NN maker)) (NP (JJ last) (NN year)) (VP (VBD sold) (NP (CD 1,214) (NNS cars)) (PP (IN in) (NP (DT the) (NNP U.S.))))))"

# Convert Lisp-style string to NLTK Tree
nltk_tree = parse_lisp_string(lisp_string)
# print(nltk_tree)
# # Since we cannot display the tree graphically, we will display it in a text format.
# nltk_tree.pretty_print()

with open(sys.argv[1], 'r', encoding="utf-8") as fin:
    data = [json.loads(line.strip()) for line in fin]
    code_res = [json.dumps({"sentence": itm["sentence"], "code": parse_lisp_string(itm["bracket"])}) for itm in data]

with open(sys.argv[2], 'w', encoding="utf-8") as fout:
    fout.write("\n".join(code_res)+"\n")
    