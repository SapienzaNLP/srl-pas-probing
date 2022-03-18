#!/bin/bash

MODELS=(\
    bert_multi_verbs_swish_en bert_multi_verbs_swish_zh \
    xlmr_verbs_swish_en xlmr_verbs_swish_zh \
    bert_nouns_identity bert_nouns_swish \
    bert_verbs_identity bert_verbs_swish \
    roberta_nouns_identity roberta_nouns_swish \
    roberta_verbs_identity roberta_verbs_swish \
)

TEST_PATHS=(\
    en/CoNLL2009_test.verbs.json zh/CoNLL2009_test.json \
    en/CoNLL2009_test.verbs.json zh/CoNLL2009_test.json \
    en/CoNLL2009_test.nouns.json en/CoNLL2009_test.nouns.json \
    en/CoNLL2009_test.verbs.json en/CoNLL2009_test.verbs.json \
    en/CoNLL2009_test.nouns.json en/CoNLL2009_test.nouns.json \
    en/CoNLL2009_test.verbs.json en/CoNLL2009_test.verbs.json \
)

len=${#MODELS[@]}

for (( i=0; i<$len; i++ )); do
    MODEL=${MODELS[$i]}
    TEST_PATH=${TEST_PATHS[$i]}
    printf "== Evaluating $MODEL ==\n"
    python evaluate_on_roleset_classification.py \
        --inventory propbank \
        --datamodule logs_probing/propbank/${MODEL}/config.json \
        --test_path data/json/${TEST_PATH} \
        --checkpoint logs_probing/propbank/${MODEL}/checkpoints/msrl*.ckpt \
        --output output/roleset_predictions/propbank/${MODEL}.txt
    printf "Done!\n\n"
done

MODELS=(\
    bert_nouns_identity_span bert_nouns_swish_span \
    bert_verbs_identity_span bert_verbs_swish_span \
)

TEST_PATHS=(\
    en/CoNLL2012_test.nouns.json en/CoNLL2012_test.nouns.json \
    en/CoNLL2012_test.verbs.json en/CoNLL2012_test.verbs.json \
)

len=${#MODELS[@]}

for (( i=0; i<$len; i++ )); do
    MODEL=${MODELS[$i]}
    TEST_PATH=${TEST_PATHS[$i]}
    printf "== Evaluating $MODEL ==\n"
    python evaluate_on_roleset_classification.py \
        --inventory propbank \
        --datamodule logs_probing/propbank/${MODEL}/config.json \
        --test_path data/json/${TEST_PATH} \
        --span_based \
        --checkpoint logs_probing/propbank/${MODEL}/checkpoints/msrl*.ckpt \
        --output output/roleset_predictions/propbank/${MODEL}.txt
    printf "Done!\n\n"
done