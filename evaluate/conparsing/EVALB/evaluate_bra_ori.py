import os
import sys
import re
import nltk
from error_process import *
def process_pred(pred_file):
    with open(pred_file,"r",encoding="utf-8") as f:
        pred_trees = [i.strip() for i in f.readlines()]
    def replace_map(parsing_tree):
        adic = {}
        with open("map.txt","r",encoding="utf-8") as f:
            data = [i.strip() for i in f.readlines()]
        for line in data:
            a,b = line.split()
            adic[b] = a
            tree = nltk.Tree.fromstring(parsing_tree)
        # Define a function to replace leaf nodes
        for pos in tree.treepositions():
            if isinstance(tree[pos], str):
                if tree[pos] in adic:
                    tree[pos] = adic[tree[pos]]
        return " ".join(str(tree).split())

    def fix_two_elements(parse_tree_str):
        pattern = r'\([^\(\)]+ [^\(\)]+ [^\(\)]+\)'
        match = re.findall(pattern, parse_tree_str)
        match_item = [i[1:-1].split() for i in match]
        ans = []
        for i in match_item:
            ans.append("("+i[0]+" "+i[1]+")"+" ("+i[0]+" "+i[2]+")")
        for i,j in zip(match,ans):
            parse_tree_str = parse_tree_str.replace(i,j)
        return parse_tree_str

    with open(pred_file.split(".txt")[0]+"_debug.txt","w",encoding="utf-8") as f:
        process_bracket_cnt = 0
        error_cnt = 0
        for i in pred_trees:
            output = i.replace("{","(").replace("}",")")
            # output = fix_two_elements(output)
            try:
                if "(" in output:
                    output = output[output.index("("):]
                    if output.count("(") == output.count(")"):
                        output = replace_map(output)
                        f.write(output+"\n")
                    elif output.count("(") > output.count(")"):
                        process_bracket_cnt += 1
                        output = replace_map(output+(output.count("(")-output.count(")"))*")")
                        f.write(output+"\n")
                    else:
                        process_bracket_cnt += 1
                        #output = replace_map(output+(output.count("(")-output.count(")"))*")")
                        output = replace_map(output[:-(output.count(")")-output.count("("))])
                        f.write(output+"\n")
                else:
                    print(output)
                    error_cnt += 1
                    f.write(""+"\n")
            except Exception as e:
                try:
                    print(output)
                    print(e)
                    error_cnt += 1
                    output = "((" + output + "))"
                    if output.count("(") == output.count(")"):
                        output = replace_map(output)
                        f.write(output+"\n")
                    elif output.count("(") > output.count(")"):
                        process_bracket_cnt += 1
                        output = replace_map(output+(output.count("(")-output.count(")"))*")")
                        f.write(output+"\n")
                    else:
                        process_bracket_cnt += 1
                        #output = replace_map(output+(output.count("(")-output.count(")"))*")")
                        output = replace_map(output[:-(output.count(")")-output.count("("))])
                        f.write(output+"\n")
                except:
                    f.write(""+"\n")
        print("error_cnt:",error_cnt)
        print("process_bracket_cnt:",process_bracket_cnt)

def processd_evaluate(pred_file,gold_file):
    def file_line_empty(file,line):
        with open(file,"r",encoding="utf-8") as f:
            datasets = f.readlines()
        datasets[line-1] = "\n"
        with open(file,"w",encoding="utf-8") as f:
            for i in datasets:
                f.write(i)
    commond  = "./evalb -p nk.prm " +gold_file+" "+pred_file +" -e 3000"
    outputs = os.popen(commond)
    outputs = "".join(outputs.read()).strip()
    results = []
    result_flag = 0
    errors = []
    skips = []
    while "Reading sentence" in outputs.split("\n")[-1]:
        commond  = "./evalb -p nk.prm " +gold_file+" "+pred_file +" -e 3000"
        outputs = os.popen(commond)
        outputs = "".join(outputs.read()).strip()
        if "Reading sentence" in outputs.split("\n")[-1]:
            lines = outputs.split("\n")
            for i in range(len(lines)):
                if "Reading sentence" in lines[i]:
                    num = i-1
                    break
            print(lines[num])
            error_sentence = outputs.split("\n")[-1]
            idx = int(error_sentence.split()[0])
            print(error_sentence)
            print(idx)
            file_line_empty(pred_file,idx)
    for line in outputs.split("\n"):
        if len(line.split()) == 5 and "=" not in line:
            errors.append(int(line.split()[0]))
            print("error:",line)
        elif len(line.split()) == 12:
            if line.split()[2] == "2":
                print("skip:",line.split()[0])
                skips.append(int(line.split()[0]))
        else:
            pass
        if line == "=== Summary ===":
            result_flag = 1
        if result_flag == 1:
            results.append(line)
        if len(line.split()) == 9:
            print("===================",line,"=================")
            temps = line.split()
            precision = float(temps[0])
            recall = float(temps[1])
            my_gold = int(temps[2])
            preds = int(temps[3])
            golds = int(temps[4])
    os.makedirs("results",exist_ok = True)
    with open("results/"+pred_file.split("/")[-1].replace(".txt","")+"_result.txt","w",encoding="utf-8") as f:
        for i in results:
            f.write(i+"\n")
    print("errors:",errors)
    print("errors number:",len(errors))
    print("skips",skips)
    print("skips number:",len(skips))
    print("error process",errors+skips)
    cnt,gold_bracket,pred_bracket = error_evaluation(errors+skips,gold_file,pred_file)
    print(cnt,gold_bracket,pred_bracket)
    my_gold += cnt
    golds += gold_bracket
    preds += pred_bracket
    precision = my_gold/preds
    recall = my_gold/golds
    f1 = 2*precision*recall/(precision+recall)
    print(f"Rectified score:\nPrecision:{precision}\nRecall:{recall}\nF1:{f1}")
    with open("results/"+pred_file.split("/")[-1].replace(".txt","")+"_result_with_error.txt","w",encoding="utf-8") as f:
        f.write("precision: "+str(my_gold/preds)+"\n")
        f.write("recall: "+str(my_gold/golds)+"\n")
        f.write("f1: "+str(2*precision*recall/(precision+recall)))
def evluate(pred_file,gold_file):
    process_pred(pred_file)
    processd_evaluate(pred_file.split(".txt")[0]+"_debug.txt",gold_file)

if __name__ == "__main__":
    pred_file = sys.argv[1]
    gold_file = sys.argv[2]
    evluate(pred_file,gold_file)
