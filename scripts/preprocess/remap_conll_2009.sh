#!/bin/bash

python3 scripts/preprocess/remap_conll_2009.py --input data/json/en/CoNLL2009_train.json --output data/json/en/CoNLL2009_train.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll_2009.py --input data/json/en/CoNLL2009_dev.json --output data/json/en/CoNLL2009_dev.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll_2009.py --input data/json/en/CoNLL2009_test.json --output data/json/en/CoNLL2009_test.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG

python3 scripts/preprocess/remap_conll_2009.py --input data/json/en/CoNLL2009_train.verbs.core.json --output data/json/en/CoNLL2009_train.va.core.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll_2009.py --input data/json/en/CoNLL2009_dev.verbs.core.json --output data/json/en/CoNLL2009_dev.va.core.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll_2009.py --input data/json/en/CoNLL2009_test.verbs.core.json --output data/json/en/CoNLL2009_test.va.core.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG