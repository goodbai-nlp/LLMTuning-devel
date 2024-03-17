from nltk import Tree

def tree_to_lisp(tree):
    """
    Converts an NLTK Tree to a Lisp-style string.
    """
    if isinstance(tree, Tree):
        children_str = ' '.join(tree_to_lisp(t) for t in tree)
        return f"({tree.label()} {children_str})"
    else:
        return tree

# Example NLTK Tree
nltk_tree = Tree('S', [Tree('NP', [Tree('PRP', ['I'])]),
                        Tree('VP', [Tree('VBD', ['saw']),
                                     Tree('NP', [Tree('DT', ['a']), Tree('NN', ['fox'])])])])

nltk_tree = Tree('TOP', [Tree('S', [Tree('NP', [Tree('DT', ['The']), Tree('JJ', ['interest-only']), Tree('NNS', ['securities'])]), Tree('VP', [Tree('VBD', ['were']), Tree('VP', [Tree('VBN', ['priced']), Tree('PP', [Tree('IN', ['at']), Tree('NP', [Tree('QP', [Tree('CD', ['35']), Tree('CD', ['1\\\\/2'])])])]), Tree('S', [Tree('VP', [Tree('TO', ['to']), Tree('VP', [Tree('VB', ['yield']), Tree('NP', [Tree('CD', ['10.72']), Tree('NN', ['%'])])])])])])]), Tree('.', ['.'])])])
nltk_tree = Tree('TOP', [Tree('S', [Tree('INTJ', [Tree('UH', ['No'])]), Tree(',', [',']), Tree('NP', [Tree('PRP', ['it'])]), Tree('VP', [Tree('VBD', ['was']), Tree('RB', ["n't"]), Tree('NP', [Tree('NNP', ['Black']), Tree('NNP', ['Monday'])])]), Tree('.', ['.'])])])
# Convert NLTK Tree to Lisp-style string
lisp_string = tree_to_lisp(nltk_tree)
print(lisp_string)