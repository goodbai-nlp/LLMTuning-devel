#!/bin/bash

BasePath=/home/export/base/ycsc_chenkh/chenkh_nvlink/online1/xfbai/
# DATA=${BasePath}/data/AMRData/LDC2017-amrparsing-llama3/test.jsonl
Gold=${BasePath}/data/AMRData/LDC2017-amrparsing-llama3/test-gold.amr
Pred=$1

python eval_amrparsing.py ${Pred} ${Gold}