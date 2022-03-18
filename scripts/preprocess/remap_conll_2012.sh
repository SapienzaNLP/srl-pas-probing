#!/bin/bash

python3 scripts/preprocess/remap_conll_2012.py --input data/json/en/CoNLL2012_train.json --output data/json/en/CoNLL2012_train.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll_2012.py --input data/json/en/CoNLL2012_dev.json --output data/json/en/CoNLL2012_dev.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll_2012.py --input data/json/en/CoNLL2012_test.json --output data/json/en/CoNLL2012_test.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG