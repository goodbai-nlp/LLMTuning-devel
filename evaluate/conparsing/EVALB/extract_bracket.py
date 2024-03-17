import nltk
from nltk.tree import Tree
import re
def remove_postags(text):
    # 用正则表达式匹配并移除词性标签
    cleaned_text = re.sub(r'\(([^)\s]+)\s([^)\s]+)\)', r'(\2)', text)
    return cleaned_text
def remove_labels_and_traces(tree, labels_to_remove):
    # 基本情况：如果树为空或者是叶节点，直接返回
    if not tree or isinstance(tree[0], str):
        return

    # 移除指定标签的节点
    tree[:] = [child for child in tree if child.label() not in labels_to_remove]

    # 对树的每一个子节点进行处理
    for index, child in enumerate(tree):
        remove_labels_and_traces(child, labels_to_remove)

    # 如果一个节点只有一个跟踪标签的子节点或者是空节点，移除该节点
    tree[:] = [child for child in tree if child or (len(child) == 1 and any(leaf not in labels_to_remove for leaf in child.leaves()))]

def extract_and_process_tree(input_string, labels_to_remove = ['TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!']):
    # 将输入字符串解析为一颗语法树
    if input_string == "":
        return []
    else:
        tree = Tree.fromstring(input_string)

        # 移除指定标签的节点
        remove_labels_and_traces(tree, labels_to_remove)
        bracket_strings = [remove_postags(str(subtree)) for subtree in tree.subtrees()]
        #bracket_strings = [str(subtree) for subtree in tree.subtrees()]

        return [" ".join(i.split()) for i in bracket_strings if "TOP" not in i and i.count("(") >= 2]