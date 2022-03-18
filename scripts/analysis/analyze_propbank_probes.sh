#!/bin/bash

MODELS=(\
    bert_multi_verbs_swish_en bert_multi_verbs_swish_zh \
    xlmr_verbs_swish_en xlmr_verbs_swish_zh \
    bert_nouns_identity bert_nouns_swish \
    bert_verbs_identity bert_verbs_swish \
    roberta_nouns_identity roberta_nouns_swish \
    roberta_verbs_identity roberta_verbs_swish \
)

len=${#MODELS[@]}

for (( i=0; i<$len; i++ )); do
    MODEL=${MODELS[$i]}
    printf "== Analyzing $MODEL ==\n"
    python analyze_weights.py \
        --inventory propbank \
        --checkpoint logs_probing/propbank/${MODEL}/checkpoints/msrl*.ckpt \
        --output output/analysis/propbank/${MODEL}.pdf
    printf "Done!\n\n"
done
