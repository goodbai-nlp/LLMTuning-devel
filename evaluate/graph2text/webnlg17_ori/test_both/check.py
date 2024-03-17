import os
our = []
refs = []
for fi in os.listdir("."):
    if fi.endswith("_eval") and fi.startswith("debug"):
        with open(fi, "r") as f:
            data = [i.strip() for i in f.readlines()]
            for i in data:
                if "( i  zmir )" in i:
                    print(fi)
                    print(i)
                    print("-----")
            our += data
    if fi.startswith("gold"):
        with open(fi, "r") as f:
            data = [i.strip() for i in f.readlines()]
            refs += data

for i,j in zip(our,refs):
    if i !=j :
        print(i)
        print(j)
        print("-----")
