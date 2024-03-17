# coding:utf-8
import json
from collections import OrderedDict

def parse_lisp_to_json(lisp_str):
    stack = []
    current_node = OrderedDict()
    for token in lisp_str.replace('(', ' ( ').replace(')', ' ) ').split():
        if token == '(':
            # Start a new node
            new_node = OrderedDict({"children": []})
            if stack:
                # Add the new node to the children of the last node in the stack
                stack[-1]["children"].append(new_node)
            stack.append(new_node)
        elif token == ')':
            # End of the current node
            current_node = stack.pop()
        else:
            # It's a word or a type
            if "type" not in stack[-1]:
                # It's the first token in the node, so it's the type
                stack[-1]["type"] = token
            else:
                # It's a word, add it to the last node
                stack[-1]["word"] = token
                stack[-1].pop("children")
    return current_node

# Sample Lisp-style string
lisp_str = "(S (NP (PRP I)) (VP (VBD saw) (NP (DT a) (NN fox))))"
lisp_str = "(TOP (S (NP (DT The) (NN luxury) (NN auto) (NN maker)) (NP (JJ last) (NN year)) (VP (VBD sold) (NP (CD 1,214) (NNS cars)) (PP (IN in) (NP (DT the) (NNP U.S.))))))"

# Convert the Lisp-style string to JSON
json_tree = parse_lisp_to_json(lisp_str)

# Convert the Python dictionary to a JSON string for display
json_str = json.dumps(json_tree, indent=4)
print(json_str)