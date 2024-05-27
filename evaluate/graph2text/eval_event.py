import re
import os
import sys
from unidecode import unidecode

def convert_text(text):
    text = ' '.join(re.split('(\W)', unidecode(text.lower())))
    text = ' '.join(text.split())
    return text

def eval_meteor_test_webnlg(folder_data, pred_file, dataset):
    cmd_string = "java -jar "+ "utils/meteor-1.5.jar " + pred_file + " " \
                  + folder_data + "/" + dataset + ".target_eval_meteor -l en -norm -r 3 > " + pred_file.replace("txt", "meteor")

    os.system(cmd_string)
    meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()
    return "meteor: "+ meteor_info.split(" ")[-1]


def eval_chrf_test_webnlg(folder_data, pred_file, dataset):

    cmd_string = "python " + "utils/chrf++.py -H " + pred_file + " -R " \
                  + folder_data + "/" + dataset + ".target_eval_crf > " + pred_file.replace("txt", "chrf")

    os.system(cmd_string)
    chrf_info_2 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[2].strip()

    return "chrf: "+ chrf_info_2.split("\t")[-1]


def eval_bleu(folder_data, pred_file, dataset):
    cmd_string = "perl " + "utils/multi-bleu.perl -lc " + folder_data + "/" + dataset + ".target_eval " \
                  + folder_data + "/" + dataset + ".target2_eval " + folder_data + "/" + dataset + ".target3_eval < " \
                  + pred_file + " > " + pred_file.replace("txt", "bleu")
    os.system(cmd_string)

    try:
        bleu_info = open(pred_file.replace("txt", "bleu"), 'r').readlines()[0].strip()
    except:
        bleu_info = -1
    pattern = r"BLEU = (\d+\.\d+)"
    match = re.search(pattern, bleu_info).group(1)
    return "bleu: "+ match


def eval_bleu_single(folder_data, pred_file, dataset):
    cmd_string = "perl " + "utils/multi-bleu.perl -lc " + folder_data + "/" + dataset + ".target_eval < " \
                  + pred_file + " > " + pred_file.replace("txt", "bleu")
    os.system(cmd_string)

    try:
        bleu_info = open(pred_file.replace("txt", "bleu"), 'r').readlines()[0].strip()
    except:
        bleu_info = -1
    pattern = r"BLEU = (\d+\.\d+)"
    match = re.search(pattern, bleu_info).group(1)
    return "bleu: "+ match


if __name__ == "__main__":
    datacate="webnlg20"
    #datacate="webnlg17"
    file = sys.argv[1]
    datacate = sys.argv[2]
    with open(file, "r") as f:
        lines = f.readlines()
    file_tok = file.replace(".txt","_tok.txt")
    with open(file_tok, "w") as f:
        for line in lines:
            f.write(convert_text(line)+"\n")
    
    print("=========BLEU=========")
    bleu_all = eval_bleu_single(f"{datacate}", file_tok, "test")
    print(f"Both: {bleu_all}")
    # print("=========Meteor=========")
    # metor_all = eval_meteor_test_webnlg(f"{datacate}", file_tok, "test")
    # print(f"Both: {metor_all}")
    print("=========Chrf++=========")
    metor_all = eval_chrf_test_webnlg(f"{datacate}", file_tok, "test")
    print(f"Both: {metor_all}")
    
    # metor_both = eval_meteor_test_webnlg("test_both",file_tok,"test_both")
    # metor_seen = eval_meteor_test_webnlg("test_seen",seen_file_tok,"test_seen")
    # metor_unseen = eval_meteor_test_webnlg("test_unseen",unseen_file_tok,"test_unseen")
    # print(f"Both: {metor_both}\nSeen: {metor_seen}\nUnseen: {metor_unseen}")
    
    # print("=========Chrf++=========")
    # chrf_both = eval_chrf_test_webnlg("test_both",file_tok,"test_both")
    # chrf_seen = eval_chrf_test_webnlg("test_seen",seen_file_tok,"test_seen")
    # chrf_unseen = eval_chrf_test_webnlg("test_unseen",unseen_file_tok,"test_unseen")
    # print(f"Both: {chr}\nSeen: {chrf_seen}\nUnseen: {chrf_unseen}")
    
