#!/bin/bash

MODELS=(\
    bert-base-last-verbs bert-base-mixed+roleset-verbs \
    bert-large-last-verbs bert-large-mixed+roleset-verbs \
)

TEST_PATHS=(\
    en/CoNLL2009_test.verbs.json en/CoNLL2009_test.verbs.json \
    en/CoNLL2009_test.verbs.json en/CoNLL2009_test.verbs.json \
)

CONLL_PATHS=(\
    en/CoNLL2009_test.txt en/CoNLL2009_test.txt \
    en/CoNLL2009_test.txt en/CoNLL2009_test.txt \
)

len=${#MODELS[@]}

for (( i=0; i<$len; i++ )); do
    MODEL=${MODELS[$i]}
    TEST_PATH=${TEST_PATHS[$i]}
    CONLL_PATH=${CONLL_PATHS[$i]}
    printf "== Evaluating $MODEL ==\n"
    python evaluate_on_srl.py \
        --datamodule logs_srl/${MODEL}/config.json \
        --checkpoint logs_srl/${MODEL}/checkpoints/msrl*.ckpt \
        --test_path data/json/${TEST_PATH} \
        --conll_test data/txt/${CONLL_PATH} \
        --gold_output output/srl_predictions/${MODEL}_gold.txt \
        --pred_output output/srl_predictions/${MODEL}_pred.txt
    printf "Done!\n\n"
done
