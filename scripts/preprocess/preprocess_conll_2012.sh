#!/bin/bash

python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_train.txt --output data/json/en/CoNLL2012_train.json --add_predicate_pos
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_dev.txt --output data/json/en/CoNLL2012_dev.json --add_predicate_pos
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_test.txt --output data/json/en/CoNLL2012_test.json --add_predicate_pos

python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_train.txt --output data/json/en/CoNLL2012_train.verbs.json --add_predicate_pos --keep_only_verb_predicates
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_dev.txt --output data/json/en/CoNLL2012_dev.verbs.json --add_predicate_pos --keep_only_verb_predicates
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_test.txt --output data/json/en/CoNLL2012_test.verbs.json --add_predicate_pos --keep_only_verb_predicates

python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_train.txt --output data/json/en/CoNLL2012_train.nouns.json --add_predicate_pos --keep_only_noun_predicates
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_dev.txt --output data/json/en/CoNLL2012_dev.nouns.json --add_predicate_pos --keep_only_noun_predicates
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_test.txt --output data/json/en/CoNLL2012_test.nouns.json --add_predicate_pos --keep_only_noun_predicates
