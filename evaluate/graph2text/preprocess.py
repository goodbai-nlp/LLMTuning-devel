# coding:utf-8

import re
import sys


def convert_text(text):
    #return text
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text


with open(sys.argv[1], 'r', encoding='utf-8') as fin:
    data = [convert_text(line.strip()) for line in fin]

with open(sys.argv[2], 'w', encoding='utf-8') as fout:
    fout.write("\n".join(data)+"\n")
