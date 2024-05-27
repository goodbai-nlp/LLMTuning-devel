# coding:utf-8
import re
import sys
import enum
import penman
import networkx as nx
from penman_interface import encode
from collections import defaultdict, Counter

BACKOFF = penman.Graph(
    [
        penman.Triple("d2", ":instance", "dog"),
        penman.Triple("b1", ":instance", "bark-01"),
        penman.Triple("b1", ":ARG0", "d2"),
    ]
)
pointer_pattern = re.compile(r'^z\d+')


def decode_amr(tokens, restore_name_ops=None):
    try:
        nodes, backreferences = decode_into_node_and_backreferences(tokens)
    except Exception as e:
        print('Decoding failure:', file=sys.stderr)
        print(e, file=sys.stderr)
        return BACKOFF, ParsedStatus.BACKOFF, (None, None)
    
    # print("Decoded nodes:", nodes)
    graph_ = graph = _fix_and_make_graph(nodes)
    # try:
    #     graph_ = graph = _fix_and_make_graph(nodes)
    #     # if collapse_name_ops:
    #     #     graph_ = graph = _split_name_ops(graph)

    # except Exception as e:
    #     print('Building failure:', file=sys.stderr)
    #     print(nodes, file=sys.stderr)
    #     print(backreferences, file=sys.stderr)
    #     print(e, file=sys.stderr)
    #     return BACKOFF, ParsedStatus.BACKOFF, (None, None)
    try:
        graph, status = connect_graph_if_not_connected(graph)
        if status == ParsedStatus.BACKOFF:
            print('Reconnection 1 failure:')
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
        return graph, status, (nodes, backreferences)
    except Exception as e:
        print('Reconnction 2 failure:', file=sys.stderr)
        print(e, file=sys.stderr)
        print(nodes, file=sys.stderr)
        print(backreferences, file=sys.stderr)
        print(graph_, file=sys.stderr)
        return BACKOFF, ParsedStatus.BACKOFF, (nodes, backreferences)


def _fix_and_make_graph(nodes, remove_pars=False):

    nodes_ = []
    for n in nodes:
        if isinstance(n, str):
            # if n.startswith('<') and n.endswith('>') and (not n.startswith('<pointer:')):
            re_res = pointer_pattern.match(n)
            if n.startswith('<') and n.endswith('>') and (re_res is None):
                pass
            else:
                nodes_.append(n)
        else:
            nodes_.append(n)
    
    nodes = nodes_
    # print("nodes in line 85", nodes)
    
    if True:
        i = 0
        nodes_ = []
        while i < len(nodes):                       # 处理<pointer>连着文本的情况
            nxt = nodes[i]
            pst = None
            if isinstance(nxt, str) and (pointer_pattern.match(nxt) is not None):
                e = pointer_pattern.match(nxt).span()[1] - 1
                if e != len(nxt) -1:
                    pst = nxt[e+1:]
                    nxt = nxt[:e+1]
                nodes_.append(nxt)
                if pst is not None:
                    nodes_.append(pst)
            else:
                nodes_.append(nxt)
            i += 1
        nodes = nodes_

        i = 1
        nodes_ = [nodes[0]]
        while i < len(nodes):
            nxt = nodes[i]
            # if isinstance(nxt, str) and nxt.startswith('<pointer:'):
            # re_res = pointer_pattern.match(nxt)
            if isinstance(nxt, str) and (pointer_pattern.match(nxt) is not None):
                nxt = 'z' + nxt[1:]           # z + {pointer number}
                fol = nodes[i+1]                # following node
                # is not expansion
                if isinstance(fol, str) and (fol.startswith(':') or (fol == ')')):
                    nodes_.append(nxt)
                else:
                    if remove_pars:
                        nodes_.append('(')
                    else:
                        if nodes_[-1] != '(':
                            nodes_.append('(')
                            #pass
                    nodes_.append(nxt)
                    nodes_.append('/')
            else:
                nodes_.append(nxt)
            i += 1
        nodes = nodes_
    
    # print("nodes in line 133", nodes)
    
    i = 0
    nodes_ = []
    while i < (len(nodes) - 1):
        if nodes[i] == ':':
            nodes_.append(nodes[i] + nodes[i+1])
            i += 2
            last = False
        else:
            nodes_.append(nodes[i])
            i += 1
            last = True
    if last:
        nodes_.append(nodes[-1])
    nodes = nodes_

    i = 0
    nodes_ = []
    while i < (len(nodes)):
        if i < 2:
            nodes_.append(nodes[i])
            i += 1
        elif nodes_[-2] == '/' and nodes[i] == '/':
            i += 2
        else:
            nodes_.append(nodes[i])
            i += 1
    nodes = nodes_

    i = 0
    newvars = 0
    variables = set()
    remap = {}
    nodes_ = []
    while i < (len(nodes)):

        next = nodes[i]

        if next == '/':
            last = nodes_[-1]
            if last in variables:
                last_remap = f"z{newvars+1000}"
                newvars += 1
                nodes_[-1] = last_remap
                remap[last] = last_remap
            variables.add(last)
            nodes_.append(next)

        elif _classify(next) == 'VAR' and next in remap and (i < len(nodes) - 1) and nodes[i+1] != '/':
            next = remap[next]
            nodes_.append(next)

        else:
            nodes_.append(next)

        i += 1

    nodes = nodes_
    pieces_ = []
    open_cnt = 0
    closed_cnt = 0
    if nodes[0] != '(':
        pieces_.append('(')
        open_cnt += 1
    for p in nodes:
        if p == '(':
            open_cnt += 1
        elif p == ')':
            closed_cnt += 1
        pieces_.append(p)
        if open_cnt == closed_cnt:
            break
    nodes = pieces_ + [')'] * (open_cnt - closed_cnt)

    pieces = []
    for piece in nodes:
        if not pieces:
            pieces.append('(')
        else:
            piece = str(piece)
            if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                piece = '"' + piece.replace('"', '') + '"'

            prev = _classify(pieces[-1])
            next = _classify(piece)

            if next == 'CONST':
                quote = False
                for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\', '_', '='):
                    if char in piece:
                        quote = True
                        break
                if quote:
                    piece = '"' + piece.strip('"') + '"'

            if  prev == '(':
                if next in ('VAR', 'I'):
                    pieces.append(piece)
            elif prev == ')':
                if next in (')', 'EDGE', 'MODE'):
                    pieces.append(piece)
            elif prev == 'VAR':
                if next in ('/', 'EDGE', 'MODE', ')'):
                    pieces.append(piece)
            elif prev == '/':
                if next in ('INST', 'I'):
                    pieces.append(piece)
            elif prev == 'INST':
                if next in (')', 'EDGE', 'MODE'):
                    pieces.append(piece)
            elif prev == 'I':
                if next in ('/', ')', 'EDGE', 'MODE'):
                    pieces.append(piece)
            elif prev == 'EDGE':
                if next in ('(', 'VAR', 'CONST', 'I'):
                    pieces.append(piece)
                elif next == ')':
                    pieces[-1] = piece
                elif next in ('EDGE', 'MODE'):
                    pieces[-1] = piece
            elif prev == 'MODE':
                if next == 'INST':
                    pieces.append(piece)
            elif prev == 'CONST':
                if next in (')', 'EDGE', 'MODE'):
                    pieces.append(piece)

    pieces_ = []
    open_cnt = 0
    closed_cnt = 0
    if pieces[0] != '(':
        pieces_.append('(')
        open_cnt += 1
    for p in pieces:
        if p == '(':
            open_cnt += 1
        elif p == ')':
            closed_cnt += 1
        pieces_.append(p)
        if open_cnt == closed_cnt:
            break
    pieces = pieces_ + [')'] * (open_cnt - closed_cnt)

    linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()

    """
    line = linearized
    # make sure parentheses match
    # copied from https://github.com/RikVN/AMR/blob/master/restoreAMR/restore_amr.py
    open_count = 0
    close_count = 0
    for i, c in enumerate(line):
        if c == '(':
            open_count += 1
        elif c == ')':
            close_count += 1
        if open_count == close_count and open_count > 0:
            line = line[:i].strip()
            break
    old_line = line
    while True:
        open_count = len(re.findall(r'\(', line))
        close_count = len(re.findall(r'\)', line))
        if open_count > close_count:
            line += ')' * (open_count - close_count)
        elif close_count > open_count:
            for i in range(close_count - open_count):
                line = line.rstrip(')')
                line = line.rstrip(' ')
        if old_line == line:
            break
        old_line = line
    """

    graph = penman.decode(linearized + ' ')
    triples = []
    newvars = 2000
    for triple in graph.triples:
        x, rel, y = triple
        if x is None:
            pass
        elif rel == ':instance' and y is None:
            triples.append(penman.Triple(x, rel, 'thing'))
        elif y is None:
            var = f'z{newvars}'
            newvars += 1
            triples.append(penman.Triple(x, rel, var))
            triples.append(penman.Triple(var, ':instance', 'thing'))
        else:
            triples.append(triple)
    graph = penman.Graph(triples)
    linearized = encode(graph)

    def fix_text(linearized=linearized):
        n = 0
        def _repl1(match):
            nonlocal n
            out = match.group(1) + match.group(2) + str(3000 + n) + ' / ' + match.group(2) + match.group(3)
            n += 1
            return out
        linearized = re.sub(r'(\(\s?)([a-z])([^\/:\)]+[:\)])', _repl1, linearized,
                            flags=re.IGNORECASE | re.MULTILINE)

        def _repl2(match):
            return match.group(1)
        linearized = re.sub(r'(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)', _repl2,
                            linearized,
                            flags=re.IGNORECASE | re.MULTILINE)

        # adds a ':' to args w/o it
        linearized = re.sub(r'([^:])(ARG)', r'\1 :\2', linearized)

        # removes edges with no node
        # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

        return linearized

    linearized = fix_text(linearized)
    g = penman.decode(linearized)
    return g

def _classify(node):
    if not isinstance(node, str):
        return "CONST"
    elif node == 'i':
        return "I"
    elif re.match(r'^[a-z]\d*$', node) is not None:
        return "VAR"
    elif node[0].isdigit():
        return "CONST"
    elif node.startswith('"') and node.endswith('"'):
        return "CONST"
    elif node in ('+', '-'):
        return "CONST"
    elif node == ':mode':
        return 'MODE'
    elif node.startswith(':'):
        return "EDGE"
    elif node in ['/', '(', ')']:
        return node
    elif node[0].isalpha():
        for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\'):
            if char in node:
                return "CONST"
        return "INST"
    else:
        return 'CONST'


def token_processing(tok):
    if tok is None:
        return None
    elif tok.isdigit():
        try:
            return eval(tok)
        except:
            return tok
    elif tok.startswith('"') and (not tok.endswith('"')):
        return tok + '"'
    elif tok.endswith('"') and (not tok.startswith('"')):
        return '"' + tok
    else:
        return tok


def decode_into_node_and_backreferences(subtokens_str):
    rex_spc = re.compile(r"<(s|/s|lit|/lit|stop|unk|pad|mask)>")
    
    subtokens = subtokens_str.split()
    
    if subtokens[0].startswith("<s>") and len(subtokens[0]) > 3:
        subtokens[0] = subtokens[0].lstrip("<s>")
        subtokens = ["<s>"] + subtokens
    
    if subtokens[-1].endswith("</s>") and len(subtokens[-1]) > 4:
        subtokens[-1] = subtokens[-1].rstrip("</s>")
        subtokens = subtokens + ["</s>"]
    
    subtokens = [str(s) for s in subtokens if s != ("<pad>")]

    tokens = []
    for itm in subtokens:
        if len(tokens) and tokens[-1] == ":":               # deal with splited relations
            tokens[-1] = tokens[-1] + itm
        elif itm.startswith("<lit>") and len(itm) > 5:      # deal with <lit> xxxx </lit>
                tokens.append("<lit>")
                tokens.append(itm[5:])
        elif itm=="-of" and tokens[-1].startswith(":") and len(tokens[-1]) > 1:
            tokens[-1] = tokens[-1] + itm
        else:
            tokens.append(itm)                              # other cases
    
    tokens = [t if t != "<unk>" else "thing" for t in tokens]

    old_tokens = tokens
    # print("tokens before <lit>", tokens)
    # <lit> Barack Obama </lit> -> "Barack Obama"
    tokens = []
    # backreferences = []
    token_to_token_map = {}
    start_search = 0
    removed = 0
    while True:
        try:

            lit_start = old_tokens.index("<lit>", start_search)
            token_addition = old_tokens[start_search:lit_start]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            tokens += token_addition

            # backreferences_addition = [
            #     token_to_token_map[b] if b > -1 else -1
            #     for b in old_backreferences[start_search:lit_start]
            # ]
            # backreferences += backreferences_addition

            lit_end = min(lit_start + 2, len(old_tokens) - 1)

            while lit_end < len(old_tokens):
                old_tok = old_tokens[lit_end]

                if isinstance(old_tok, str) and (
                    (old_tok.startswith(":") and len(old_tok) > 3) or (old_tok == "<stop>")
                ):
                    res_tok = old_tokens[lit_start + 1 : lit_end]
                    for i in range(lit_start, lit_end):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1 : lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + "_".join(res) + '"'

                    removed += len(res_tok)
                    start_search = lit_end
                    tokens += [res, old_tok]
                    # backreferences += [-1, -1]
                    break

                elif old_tok == "</lit>":
                    res_tok = old_tokens[lit_start + 1 : lit_end]
                    for i in range(lit_start, lit_end + 1):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1 : lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + "_".join(res) + '"'

                    removed += len(res_tok) + 1
                    start_search = lit_end + 1
                    tokens.append(res)
                    # backreferences.append(-1)
                    break

                else:
                    lit_end += 1
                    start_search = lit_end

        except ValueError:
            token_addition = old_tokens[start_search:]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            # backreferences_addition = [
            #     token_to_token_map[b] if b > -1 else b for b in old_backreferences[start_search:]
            # ]
            tokens += token_addition
            # backreferences += backreferences_addition
            break

    tokens = [token_processing(t) for t in tokens]

    shift = 1
    if tokens[1] == "<s>":
        shift = 2

    tokens = tokens[shift:]

    if tokens[-1] == "</s>":
        tokens.pop()
        # backreferences.pop()

    return tokens, None


def decode_into_node_and_backreferences_ori(subtoken_ids, tokenizer):
    rex_arg = re.compile(f"^{tokenizer.INIT}(op|snt|conj|prep)")
    rex_spc = re.compile(r"<(s|/s|lit|/lit|stop|unk|pad|mask)>")
    
    # subtoken_ids.insert(1,36)           # add "(" id
    # subtoken_ids.insert(-1, 4839)       # add ")" id

    # get strings
    # subtokens = [tokenizer.decoder.get(t) for t in subtoken_ids]
    subtokens = tokenizer.convert_ids_to_tokens(subtoken_ids)
    # subtokens_new = tokenizer.batch_decode([subtoken_ids])[0]
    # subtokens = tokenizer.decode(subtoken_ids).split()
    # print("subtokens:", subtokens)
    # print("subtokens_new:", subtokens_new.split())
    # exit()
    # fix backreferences
    # print(f"len(tokenizer): {len(tokenizer)}")
    # print(f"len(tokenizer.encoder): {len(tokenizer.encoder)}")
    # exit()
    subtoken_backreferences = [max(t - len(tokenizer), -1) for t in subtoken_ids]
    # print("subtoken_backreferences", subtoken_backreferences)
    # strip padding
    subtokens, subtoken_backreferences = zip(
        *[
            (s, b)
            for s, b in zip(subtokens, subtoken_backreferences)
            if s != ("<pad>")
        ]
    )

    # subword collapse
    tokens = []
    backreferences = []
    subword_to_token_map = {}
    current_token_i = 0
    for subw_i, (subw_backr, subtok) in enumerate(zip(subtoken_backreferences, subtokens)):
        subword_to_token_map[subw_i] = current_token_i

        # if empty you cannot do anything but add a new word
        if not tokens:
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # backref can't be splitted
        elif subw_backr > -1:
            tokens.append(None)
            backreferences.append(subword_to_token_map[subw_backr])
            current_token_i += 1

        # after a special token release  
        elif isinstance(tokens[-1], str) and rex_spc.match(tokens[-1]):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # after a subtoken ':' (which should be followed by the rest of the edge) ignore tokenizer.INIT
        # TODO: this is an ugly patch due to the fact that BART tokenizer splits after ':'
        elif (tokens[-1] == ":") and rex_arg.match(subtok):
            tokens[-1] = tokens[-1] + subtok[1:]

        # leading tokenizer.INIT
        elif subtok.startswith(tokenizer.INIT):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # very ugly patch for some cases in which tokenizer.INIT is not in the following token to the edge
        elif (
            isinstance(tokens[-1], str)
            and tokens[-1].startswith(":")
            and tokens[-1][-1].isdigit()
            and (subtok != "-of")
        ):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # in any other case attach to the previous
        else:
            tokens[-1] = tokens[-1] + subtok

    # print("token concatenated:", tokens)
    
    # strip INIT and fix byte-level
    tokens = [
        tokenizer.convert_tokens_to_string(list(t)).lstrip() if isinstance(t, str) else t
        for t in tokens
    ]
    # tokens = [t.replace(tokenizer.INIT, '') if isinstance(t, str) else t for t in tokens]

    # unks are substituted with thing
    tokens = [t if t != "<unk>" else "thing" for t in tokens]

    old_tokens = tokens
    old_backreferences = backreferences

    # <lit> Barack Obama </lit> -> "Barack Obama"
    tokens = []
    backreferences = []
    token_to_token_map = {}
    start_search = 0
    removed = 0
    while True:
        try:

            lit_start = old_tokens.index("<lit>", start_search)
            token_addition = old_tokens[start_search:lit_start]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            tokens += token_addition

            backreferences_addition = [
                token_to_token_map[b] if b > -1 else -1
                for b in old_backreferences[start_search:lit_start]
            ]
            backreferences += backreferences_addition

            lit_end = min(lit_start + 2, len(old_tokens) - 1)

            while lit_end < len(old_tokens):
                old_tok = old_tokens[lit_end]

                if isinstance(old_tok, str) and (
                    (old_tok.startswith(":") and len(old_tok) > 3) or (old_tok == "<stop>")
                ):
                    res_tok = old_tokens[lit_start + 1 : lit_end]
                    for i in range(lit_start, lit_end):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1 : lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + "_".join(res) + '"'

                    removed += len(res_tok)
                    start_search = lit_end
                    tokens += [res, old_tok]
                    backreferences += [-1, -1]
                    break

                elif old_tok == "</lit>":
                    res_tok = old_tokens[lit_start + 1 : lit_end]
                    for i in range(lit_start, lit_end + 1):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1 : lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + "_".join(res) + '"'

                    removed += len(res_tok) + 1
                    start_search = lit_end + 1
                    tokens.append(res)
                    backreferences.append(-1)
                    break

                else:
                    lit_end += 1
                    start_search = lit_end

        except ValueError:
            token_addition = old_tokens[start_search:]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            backreferences_addition = [
                token_to_token_map[b] if b > -1 else b for b in old_backreferences[start_search:]
            ]
            tokens += token_addition
            backreferences += backreferences_addition
            break

    tokens = [token_processing(t) for t in tokens]

    shift = 1
    if tokens[1] == "<s>":
        shift = 2

    tokens = tokens[shift:]
    backreferences = [b if b == -1 else b - shift for b in backreferences[shift:]]

    if tokens[-1] == "</s>":
        tokens.pop()
        backreferences.pop()

    return tokens, backreferences


def index_of(element, iterable, default=None, start=None, end=None):
    if not callable(element):

        def check(x):
            return element == x

    else:
        check = element
    if start is None:
        start = 0
    if end is None:
        end = len(iterable)
    item = start
    while item < end:
        if check(iterable[item]):
            return item
        item += 1
    return default


def separate_edges_nodes(edges_nodes_slice, *other):
    is_arg = lambda x: isinstance(x, str) and x.startswith(":")
    start = 0
    edges = []
    nodes = []
    l = len(edges_nodes_slice)
    while start < l:
        edge_index = index_of(is_arg, edges_nodes_slice, start=start)
        if edge_index is None or edge_index == (l - 1):
            break
        if is_arg(edges_nodes_slice[edge_index + 1]):
            start = edge_index + 1
            continue
        edges.append(edge_index)
        nodes.append(edge_index + 1)
        start = edge_index + 2
    ret = []
    for oth in other:
        edges_oth = [oth[i] for i in edges]
        nodes_oth = [oth[i] for i in nodes]
        ret.append((edges_oth, nodes_oth))
    return ret


def _split_name_ops(graph):
    # identify name triples
    name_vars = {}
    for i, (v1, rel, v2) in enumerate(graph.triples):
        if rel == ":instance" and v2 == "name":
            name_vars[v1] = 1

    # check if they have ops
    name_vars_to_ops = defaultdict(list)
    for i, (v1, rel, v2) in enumerate(graph.triples):
        if v1 in name_vars and rel.startswith(":op"):
            name_vars_to_ops[v1].append((i, rel, v2.strip('"')))

    triples = graph.triples.copy()
    for nv, ops in name_vars_to_ops.items():
        ops = sorted(ops, key=lambda x: int(x[1][3:]))
        idx, _, lits = zip(*ops)
        for i in idx:
            triples[i] = None

        lits = ['"' + l + '"' for lit in lits for l in lit.split("_")]

        tt = []
        for i, l in enumerate(lits, start=1):
            rel = ":op" + str(i)
            tt.append(penman.Triple(nv, rel, l))

        triples[min(idx)] = tt

    triples = [t if isinstance(t, list) else [t] for t in triples if t is not None]
    triples = [t for tt in triples for t in tt]

    graph_ = penman.Graph(triples)
    graph_.metadata = graph.metadata
    return graph_


def _reconstruct_graph_from_nodes(nodes, backreferences):
    triples = []
    triples_added = set()

    variable2index = {}
    index2variable = {}
    start_index = 0

    cnt = defaultdict(Counter)

    while start_index < len(nodes):
        stop_index = index_of("<stop>", nodes, default=len(nodes) + 1, start=start_index)
        old_start_index = start_index
        start_index = stop_index + 1

        src_node, src_backr = nodes[old_start_index], backreferences[old_start_index]

        if src_node == "<stop>":
            continue

        trg_nodes_edges = nodes[old_start_index:stop_index]
        trg_nodes_edges_backr = backreferences[old_start_index:stop_index]
        trg_nodes_edges_indices = list(range(old_start_index, stop_index))

        if isinstance(src_node, str):
            if src_node in ("<s>", "</s>", "<stop>"):
                continue
            elif ("/" in src_node) or (":" in src_node) or ("(" in src_node) or (")" in src_node):
                src_node = "thing"

        if src_node is not None:
            src_node = str(src_node)
            src_var = src_node[0].lower()
            if not src_var not in "abcdefghijklmnopqrstuvwxyz":
                src_var = "x"
            # src_var = f'{src_var}_{len(variable2index)}'
            src_var = f"{src_var}{len(variable2index)}"
            src_var_i = old_start_index
            variable2index[src_var] = src_var_i
            index2variable[src_var_i] = src_var
            triple = penman.Triple(src_var, ":instance", src_node)
            if triple not in triples_added:
                triples.append(triple)
                triples_added.add(triple)
        else:
            if src_backr in index2variable:
                src_var = index2variable[src_backr]
        # more resilient logic here
        (trg_edges, trg_nodes), (_, trg_nodes_backr), (_, trg_nodes_indices) = separate_edges_nodes(
            trg_nodes_edges, trg_nodes_edges, trg_nodes_edges_backr, trg_nodes_edges_indices
        )

        for n, e, nb, ni in zip(trg_nodes, trg_edges, trg_nodes_backr, trg_nodes_indices):

            if isinstance(n, str) and n.startswith(":"):
                continue
            if isinstance(n, str) and n.startswith("<") and n.endswith(">"):
                continue
            if e == ":li":
                pass
            elif len(e) < 4 or (not e.startswith(":")):
                continue

            # same edge more than once
            num = cnt[src_var][e]
            # num = 0
            if num:

                if e.startswith(":op") or e.startswith(":snt"):
                    continue
                # elif e.startswith(':ARG'):
                #    continue
                elif num > 3:
                    continue

            if n is None:
                if nb not in index2variable:
                    continue
                trg_var = index2variable[nb]
                trg = trg_var
            elif e == ":mode":
                trg = n
            elif (
                (not isinstance(n, str))
                or re.match(r"^[+-]?\d+\.?\d*$", n)
                or (n == "-")
                or (n == "+")
            ):
                trg = str(n)
            elif n.startswith('"') and n.endswith('"') and len(n) > 2:
                trg = '"' + n.replace('"', "") + '"'
            elif ("/" in n) or (":" in n) or ("(" in n) or (")" in n) or ("=" in n):
                trg = f'"{n}"'
            elif n == '"':
                continue
            elif (
                (n.startswith('"') and (not n.endswith('"')))
                or (not n.startswith('"') and (n.endswith('"')))
                or ('"' in n)
            ):
                trg = '"' + n.replace('"', "") + '"'
            else:
                trg_var = n[0].lower()
                if trg_var not in "abcdefghijklmnopqrstuvwxyz":
                    trg_var = "x"
                # trg_var = f'{trg_var}_{len(variable2index)}'
                trg_var = f"{trg_var}{len(variable2index)}"
                trg_var_i = ni
                variable2index[trg_var] = trg_var_i
                index2variable[trg_var_i] = trg_var
                triple = penman.Triple(trg_var, ":instance", n)
                if triple not in triples_added:
                    triples.append(triple)
                    triples_added.add(triple)
                trg = trg_var

            triple = penman.Triple(src_var, e, trg)
            if triple not in triples_added:
                triples.append(triple)
                triples_added.add(triple)

            cnt[src_var][e] += 1

    return penman.Graph(triples)


def build_graph(nodes, backreferences, restore_name_ops=False):
    graph = _reconstruct_graph_from_nodes(nodes, backreferences)
    if restore_name_ops:
        graph = _split_name_ops(graph)
    return graph


class ParsedStatus(enum.Enum):
    OK = 0
    FIXED = 1
    BACKOFF = 2


def connect_graph_if_not_connected(graph):

    try:
        encoded = encode(graph)
        return graph, ParsedStatus.OK
    except:
        pass

    nxgraph = nx.MultiGraph()
    variables = graph.variables()
    for v1, _, v2 in graph.triples:
        if v1 in variables and v2 in variables:
            nxgraph.add_edge(v1, v2)
        elif v1 in variables:
            nxgraph.add_edge(v1, v1)

    triples = graph.triples.copy()
    new_triples = []
    addition = f"a{len(variables) + 1}"
    triples.append(penman.Triple(addition, ":instance", "and"))
    for i, conn_set in enumerate(nx.connected_components(nxgraph), start=1):
        edge = f":op{i}"
        conn_set = sorted(conn_set, key=lambda x: int(x[1:]))
        conn_set = [c for c in conn_set if c in variables]
        node = conn_set[0]
        new_triples.append(penman.Triple(addition, edge, node))
    triples = new_triples + triples
    metadata = graph.metadata
    graph = penman.Graph(triples)
    graph.metadata.update(metadata)
    encode(graph)

    return graph, ParsedStatus.FIXED


def restore_backreferences_from_pointers(nodes):
    new_nodes, new_backreferences = [], []
    prev_pointer = None
    pointer2i = {}
    for n in nodes:
        # is_pointer = isinstance(n, str) and n.startswith("<pointer:") and n.endswith(">")
        is_pointer = isinstance(n, str) and (pointer_pattern.match(nxt) is not None)
        
        if not is_pointer:
            if prev_pointer is not None:
                if prev_pointer in pointer2i:
                    new_nodes.append(None)
                    new_backreferences.append(pointer2i[prev_pointer])
                    new_nodes.append(n)
                    new_backreferences.append(-1)

                else:
                    pointer2i[prev_pointer] = len(new_nodes)
                    new_nodes.append(n)
                    new_backreferences.append(-1)
            else:
                new_nodes.append(n)
                new_backreferences.append(-1)

            prev_pointer = None
        else:
            prev_pointer = n
    return new_nodes, new_backreferences