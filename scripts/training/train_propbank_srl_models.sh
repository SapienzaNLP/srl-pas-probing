#!/bin/bash

python train_on_srl.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --batch_size 32 \
    --inventory propbank \
    --language_model_type bert_embeddings \
    --language_model_name bert-base-cased

python train_on_srl.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --batch_size 32 \
    --inventory propbank \
    --language_model_type mixed_bert_embeddings \
    --language_model_name bert-base-cased

python train_on_srl.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --batch_size 32 \
    --inventory propbank \
    --language_model_type mixed_bert_embeddings \
    --language_model_name bert-base-cased \
    --use_roleset_encodings



python train_on_srl.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --batch_size 32 \
    --inventory propbank \
    --language_model_type bert_embeddings \
    --language_model_name bert-large-cased

python train_on_srl.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --batch_size 32 \
    --inventory propbank \
    --language_model_type mixed_bert_embeddings \
    --language_model_name bert-large-cased

python train_on_srl.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --batch_size 32 \
    --inventory propbank \
    --language_model_type mixed_bert_embeddings \
    --language_model_name bert-large-cased \
    --use_roleset_encodings
