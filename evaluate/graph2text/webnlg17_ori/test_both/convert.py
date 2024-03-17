import os
import re
from unidecode import unidecode
def convert_text(text):
    #return text
    text = unidecode(text.lower())
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

for fi in os.listdir("."):
    if fi.endswith("_eval"):
        with open(fi, "r") as f:
            data = [i.strip() for i in f.readlines()]
        with open("debug_"+fi, "w") as f:
            for i in data:
                f.write(convert_text(i)+"\n")
# print("the atatürk monument ( i̇zmir ) is found in turkey , where the capital is ankara and the leader is ahmet davutoğlu .")
# print(convert_text("the atatürk monument ( i̇zmir ) is found in turkey , where the capital is ankara and the leader is ahmet davutoğlu ."))
# print("pietro canonica designed the ataturk monument ( pietro canonica designed the ataturk monument ( izmir ) which was inaugurated in turkey on 27 july 1932 . turkey is led by the president of turkey .zmir ) which was inaugurated in turkey on 27 july 1932 . turkey is led by the president of turkey .")
# print(convert_text("pietro canonica designed the ataturk monument ( izmir ) which was inaugurated in turkey on 27 july 1932 . turkey is led by the president of turkey ."))