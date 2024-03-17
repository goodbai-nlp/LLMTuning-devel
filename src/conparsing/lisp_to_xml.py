# coding:utf-8
from nltk import Tree

def parse_lisp_string(lisp_string):
    """
    Parses a Lisp-style string and returns an NLTK Tree.
    """
    tokens = lisp_string.replace('(', ' ( ').replace(')', ' ) ').split()
    
    def parse(tokens):
        if len(tokens) == 0:
            raise ValueError('Unexpected EOF while reading')
        token = tokens.pop(0)
        if token == '(':
            children = []
            while tokens[0] != ')':
                children.append(parse(tokens))
            tokens.pop(0)                   # pop off ')'
            return Tree(children[0], children[1:])
        else:
            return token

    return parse(tokens)

# Example Lisp-style string
lisp_string = "(S (NP (PRP I)) (VP (VBD saw) (NP (DT a) (NN fox))))"

# Convert Lisp-style string to NLTK Tree
nltk_tree = parse_lisp_string(lisp_string)
# Since we cannot display the tree graphically, we will display it in a text format.
nltk_tree.pretty_print()
