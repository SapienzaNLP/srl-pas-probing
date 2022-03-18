#!/bin/bash

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_train.txt --output data/json/en/CoNLL2009_train.json --add_predicate_pos --keep_pos_tags
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_dev.txt --output data/json/en/CoNLL2009_dev.json --add_predicate_pos --keep_pos_tags
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test.txt --output data/json/en/CoNLL2009_test.json --add_predicate_pos --keep_pos_tags
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test_ood.txt --output data/json/en/CoNLL2009_test_ood.json --add_predicate_pos --keep_pos_tags

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_train.txt --output data/json/en/CoNLL2009_train.verbs.json --add_predicate_pos --keep_only_verb_predicates
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_dev.txt --output data/json/en/CoNLL2009_dev.verbs.json --add_predicate_pos --keep_only_verb_predicates
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test.txt --output data/json/en/CoNLL2009_test.verbs.json --add_predicate_pos --keep_only_verb_predicates
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test_ood.txt --output data/json/en/CoNLL2009_test_ood.verbs.json --add_predicate_pos --keep_only_verb_predicates

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_train.txt --output data/json/en/CoNLL2009_train.nouns.json --add_predicate_pos --keep_only_noun_predicates
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_dev.txt --output data/json/en/CoNLL2009_dev.nouns.json --add_predicate_pos --keep_only_noun_predicates
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test.txt --output data/json/en/CoNLL2009_test.nouns.json --add_predicate_pos --keep_only_noun_predicates
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test_ood.txt --output data/json/en/CoNLL2009_test_ood.nouns.json --add_predicate_pos --keep_only_noun_predicates

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/zh/CoNLL2009_train.txt --output data/json/zh/CoNLL2009_train.json --keep_pos_tags
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/zh/CoNLL2009_dev.txt --output data/json/zh/CoNLL2009_dev.json --keep_pos_tags
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/zh/CoNLL2009_test.txt --output data/json/zh/CoNLL2009_test.json --keep_pos_tags
