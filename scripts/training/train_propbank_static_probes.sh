# RoBERTa-base / verbs / identity
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --language_model_name roberta-base \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation identity \
    --return_static_embeddings

# RoBERTa-base / verbs / swish
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --language_model_name roberta-base \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings

# RoBERTa-base / nouns / identity
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.nouns.json \
    --dev_path data/json/en/CoNLL2009_dev.nouns.json \
    --test_path data/json/en/CoNLL2009_test.nouns.json \
    --language_model_name roberta-base \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation identity \
    --return_static_embeddings

# RoBERTa-base / nouns / swish
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.nouns.json \
    --dev_path data/json/en/CoNLL2009_dev.nouns.json \
    --test_path data/json/en/CoNLL2009_test.nouns.json \
    --language_model_name roberta-base \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings

# BERT-base / verbs / identity
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation identity \
    --return_static_embeddings

# BERT-base / verbs / swish
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings

# BERT-base / nouns / identity
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.nouns.json \
    --dev_path data/json/en/CoNLL2009_dev.nouns.json \
    --test_path data/json/en/CoNLL2009_test.nouns.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation identity \
    --return_static_embeddings

# BERT-base / nouns / swish
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.nouns.json \
    --dev_path data/json/en/CoNLL2009_dev.nouns.json \
    --test_path data/json/en/CoNLL2009_test.nouns.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings

# BERT-base / verbs / identity / span
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2012_train.verbs.json \
    --dev_path data/json/en/CoNLL2012_dev.verbs.json \
    --test_path data/json/en/CoNLL2012_test.verbs.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation identity \
    --span_based \
    --return_static_embeddings

# BERT-base / verbs / swish / span
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2012_train.verbs.json \
    --dev_path data/json/en/CoNLL2012_dev.verbs.json \
    --test_path data/json/en/CoNLL2012_test.verbs.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --span_based \
    --return_static_embeddings

# BERT-base / nouns / identity / span
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2012_train.nouns.json \
    --dev_path data/json/en/CoNLL2012_dev.nouns.json \
    --test_path data/json/en/CoNLL2012_test.nouns.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation identity \
    --span_based \
    --return_static_embeddings

# BERT-base / nouns / swish / span
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2012_train.nouns.json \
    --dev_path data/json/en/CoNLL2012_dev.nouns.json \
    --test_path data/json/en/CoNLL2012_test.nouns.json \
    --language_model_name bert-base-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --span_based \
    --return_static_embeddings

# BERT-multi / verbs / swish / en
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --language_model_name bert-base-multilingual-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings

# BERT-multi / verbs / swish / zh
python train_on_roleset_classification.py \
    --train_path data/json/zh/CoNLL2009_train.json \
    --dev_path data/json/zh/CoNLL2009_dev.json \
    --test_path data/json/zh/CoNLL2009_test.json \
    --language_model_name bert-base-multilingual-cased \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings

# BERT-multi / verbs / swish / en
python train_on_roleset_classification.py \
    --train_path data/json/en/CoNLL2009_train.verbs.json \
    --dev_path data/json/en/CoNLL2009_dev.verbs.json \
    --test_path data/json/en/CoNLL2009_test.verbs.json \
    --language_model_name xlm-roberta-base \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings

# BERT-multi / verbs / swish / zh
python train_on_roleset_classification.py \
    --train_path data/json/zh/CoNLL2009_train.json \
    --dev_path data/json/zh/CoNLL2009_dev.json \
    --test_path data/json/zh/CoNLL2009_test.json \
    --language_model_name xlm-roberta-base \
    --batch_size 32 \
    --max_epochs 20 \
    --language_model_type mixed_bert_embeddings \
    --inventory propbank \
    --word_encoding_activation swish \
    --return_static_embeddings
