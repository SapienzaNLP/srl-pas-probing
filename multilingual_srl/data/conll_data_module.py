import json
import os

from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from multilingual_srl.data.conll_dataset import ConllDataset


class ConllDataModule(pl.LightningDataModule):

    _UNKNOWN_TOKEN = '<UNK>'
    _PADDING_LABEL_ID = -1

    def __init__(
        self,
        inventory='propbank',
        span_based=False,
        train_path=None,
        dev_path=None,
        test_path=None,
        language_model_type='bert_embeddings',
        language_model_name='bert-base-multilingual-cased',
        batch_size=32,
        num_workers=8,
    ):
        super().__init__()

        self.inventory = inventory
        self.span_based = span_based

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.language_model_type = language_model_type
        self.language_model_name = language_model_name

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.unknown_token = ConllDataModule._UNKNOWN_TOKEN
        self.padding_label_id = ConllDataModule._PADDING_LABEL_ID

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = ConllDataset(self.train_path, span_based=self.span_based)
            self.dev_data = ConllDataset(self.dev_path, span_based=self.span_based)
            self._build_output_maps()
            self.steps_per_epoch = len(self.train_data) // self.batch_size
        
        if stage == 'test' or stage is None:
            self.test_data = ConllDataset(self.test_path, span_based=self.span_based)
        
        self._setup_tokenizer()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_sentences)

    def val_dataloader(self):
        return DataLoader(
            self.dev_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_sentences)

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_sentences)

    def encode_sentence(self, sentence):
        '''
            Given a sentence object, returns an input sample.
        '''
        # subword_indices is a list of incrementing indices, one for each (sub)token.
        # Each index indicates the index in the original sentence of the word the subtoken belongs to.
        # Ex: ['This', 'is', 'an', 'em', '##bed', '##ding', '.']
        #     [0,      1,     2,   3,    3,       3,        4]
        subword_indices = []

        # Length of the sentence in words.
        sequence_length = len(sentence['words'])
        # Add [CLS] and [SEP] to the original length.
        sequence_length += 2

        if self.language_model_type == 'bert_embeddings' or self.language_model_type == 'mixed_bert_embeddings':
            model_inputs = self.tokenizer(sentence['words'], is_split_into_words=True, return_tensors='pt')
            subword_indices = model_inputs.word_ids()
            subword_indices[0] = 0
            subword_indices[-1] = max(subword_indices[1:-1]) + 2
            subword_indices[1:-1] = [idx + 1 for idx in subword_indices[1:-1]]
            subword_indices = torch.as_tensor(subword_indices)
        
        else:
            raise ValueError('Unsupported value for input_representation: {}'.format(self.language_model_type))

        return {
            'model_inputs': model_inputs,
            'subword_indices': subword_indices,
            'sequence_length': sequence_length,
        }

    def encode_labels(self, sentence):
        '''
            Given a sentence object, returns its label ids (senses and roles).
        '''

        sentence_length = len(sentence['words'])

        # List of 0s and 1s to indicate the predicates in the input sentence.
        predicates = []
        # List of sense ids.
        senses = []
        # List of predicate indices.
        predicate_indices = []
        # List of lists of 0.0/1.0 depending on whether the predicate-role couple appears in the sentence.
        rolesets = []
        # List of lists of role ids (n x n, where n is the length of the sentence).
        roles = [[self.padding_label_id] * sentence_length] * sentence_length

        for word_index, predicate in enumerate(sentence['predicates']):
            if predicate != '_':
                predicate_indices.append(word_index)
                predicates.append(1)
            else:
                predicates.append(0)

            if predicate in self.sense2id:
                senses.append(self.sense2id[predicate])
            elif predicate != '_':
                senses.append(self.unknown_sense_id)

        for predicate_index, predicate_roles in sentence['roles'].items():
            predicate_role_ids = []
            for role in predicate_roles:
                if role in self.role2id:
                    predicate_role_ids.append(self.role2id[role])
                else:
                    predicate_role_ids.append(self.unknown_role_id)
            roles[predicate_index] = predicate_role_ids

        for predicate_index, predicate_roleset in sentence['rolesets'].items():
            predicate_role_values = [0.0] * len(self.role2id)
            for role in predicate_roleset:
                if role in self.role2id:
                    predicate_role_values[self.role2id[role]] = 1.0

            rolesets.append(predicate_role_values)

        return {
            'predicate_indices': predicate_indices,
            'predicates': torch.as_tensor(predicates),
            'senses': torch.as_tensor(senses),
            'roles': roles,
            'rolesets': rolesets,
        }
    
    def decode(self, x, y):
        '''
            Given a sample and its label ids (in a batch), returns the labels.
        '''
        word_ids = x['word_ids']
        sentence_lengths = x['sequence_lengths']
        predicate_indices = list(map(list, zip(*x['predicate_indices'])))

        predicates = []
        if 'predicates' in y:
            predicate_ids = torch.argmax(y['predicates'], dim=-1).tolist()
            for sentence_predicate_ids, sentence_length in zip(predicate_ids, sentence_lengths):
                sentence_predicate_ids = sentence_predicate_ids[:sentence_length - 2]
                predicates.append([p for p in sentence_predicate_ids])

        senses = {}
        if 'senses' in y:
            sense_ids = torch.argmax(y['senses'], dim=-1).tolist()
            for (sentence_index, predicate_index), sense_id in zip(predicate_indices, sense_ids):
                if sentence_index not in senses:
                    senses[sentence_index] = {}
                senses[sentence_index][predicate_index] = self.id2sense[sense_id]

        roles = {i: {} for i in range(len(word_ids))}
        if 'roles' in y:
            role_ids = torch.argmax(y['roles'], dim=-1).tolist()
            for (sentence_index, predicate_index), predicate_role_ids in zip(predicate_indices, role_ids):
                sentence_length = sentence_lengths[sentence_index]
                predicate_role_ids = predicate_role_ids[:sentence_length - 2]
                predicate_roles = [self.id2role[r] for r in predicate_role_ids]
                roles[sentence_index][predicate_index] = predicate_roles

        rolesets = {}
        if 'rolesets' in y:
            roleset_scores = torch.sigmoid(y['rolesets']).tolist()
            for (sentence_index, predicate_index), predicate_roles in zip(predicate_indices, roleset_scores):
                if sentence_index not in rolesets:
                    rolesets[sentence_index] = {}
                rolesets[sentence_index][predicate_index] = []
                for role_index, score in enumerate(predicate_roles):
                    if score > 0.5:
                        rolesets[sentence_index][predicate_index].append(self.id2role[role_index])

        return {
            'predicates': predicates,
            'senses': senses,
            'roles': roles,
            'rolesets': rolesets,
            
        }
    
    def _setup_tokenizer(self):
        if self.language_model_type == 'elmo_embeddings':
            self.padding_token_id = 0
            self.unknown_token_id = 0

        elif self.language_model_type == 'bert_embeddings' or self.language_model_type == 'mixed_bert_embeddings':
            self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name, add_prefix_space=True)
            self.padding_token_id = self.tokenizer.pad_token_id
            self.unknown_token_id = self.tokenizer.unk_token_id

        else:
            raise ValueError('Unsupported value for input_representation: {}'.format(self.language_model_type))
    
    def _build_output_maps(self):
        self.sense2id = { self.unknown_token: 0 }
        if self.inventory == 'propbank':
            if self.span_based:
                self.role2id = {
                    '_': 0,
                    self.unknown_token: 1,
                    'A0': 2,
                    'A1': 3,
                    'A2': 4,
                    'A3': 5,
                    'A4': 6,
                    'A5': 7,
                }
            else:
                self.role2id = {
                    '_': 0,
                    self.unknown_token: 1,
                    'B-ARG0': 2,
                    'B-ARG1': 3,
                    'B-ARG2': 4,
                    'B-ARG3': 5,
                    'B-ARG4': 6,
                    'B-ARG5': 7,
                }
        elif self.inventory == 'verbatlas':
            self.role2id = {
                self.unknown_token: 0,
                '_': 1,
                'Agent': 2,
                'Asset': 3,
                'Attribute': 4,
                'Beneficiary': 5,
                'Cause': 6,
                'Co-Agent': 7,
                'Co-Patient': 8,
                'Co-Theme': 9,
                'Destination': 10,
                'Experiencer': 11,
                'Extent': 12,
                'Goal': 13,
                'Idiom': 14,
                'Instrument': 15,
                'Location': 16,
                'Material': 17,
                'Patient': 18,
                'Product': 19,
                'Purpose': 20,
                'Recipient': 21,
                'Result': 22,
                'Source': 23,
                'Stimulus': 24,
                'Theme': 25,
                'Time': 26,
                'Topic': 27,
                'Value': 28
            }
        else:
            raise ValueError('Unsupported value for inventory: {}'.format(self.inventory))

        for sentence in self.train_data:

            for sense in sentence['predicates']:
                if sense not in self.sense2id and sense != '_':
                    self.sense2id[sense] = len(self.sense2id)

            for roles in sentence['roles'].values():
                for role in roles:
                    if role != '_' and role not in self.role2id:
                        self.role2id[role] = len(self.role2id)

        self.id2sense = {id: predicate for predicate, id in self.sense2id.items()}
        self.id2role = {id: role for role, id in self.role2id.items()}

        self.unknown_sense_id = self.sense2id[self.unknown_token]
        self.unknown_role_id = self.role2id[self.unknown_token]

        self.num_senses = len(self.sense2id)
        self.num_roles = len(self.role2id)
    
    def _collate_sentences(self, sentences):
        batched_x = {
            'sentence_ids': [],
            'predicate_indices': [[], []],

            'word_ids': [],
            'attention_mask': [],
            'subword_indices': [],
            'sequence_lengths': [],
        }

        batched_y = {
            'predicates': [],
            'senses': [],
            'roles': [],
            'rolesets': [],
        }

        max_sequence_length = 0
        for sentence_index, sentence in enumerate(sentences):
            encoded_sentence = self.encode_sentence(sentence)
            encoded_labels = self.encode_labels(sentence)

            max_sequence_length = max(encoded_sentence['sequence_length'], max_sequence_length)

            batched_x['sentence_ids'].append(sentence['sentence_id'])
            batched_x['predicate_indices'][0].extend([sentence_index]*len(encoded_labels['predicate_indices']))
            batched_x['predicate_indices'][1].extend(encoded_labels['predicate_indices'])

            batched_x['word_ids'].append(encoded_sentence['model_inputs']['input_ids'].squeeze())
            batched_x['attention_mask'].append(encoded_sentence['model_inputs']['attention_mask'].squeeze())
            batched_x['subword_indices'].append(encoded_sentence['subword_indices'])
            batched_x['sequence_lengths'].append(encoded_sentence['sequence_length'])

            batched_y['predicates'].append(encoded_labels['predicates'])
            batched_y['senses'].extend(encoded_labels['senses'])
            batched_y['rolesets'].extend(encoded_labels['rolesets'])
            batched_y['roles'].append(encoded_labels['roles'])

        if self.language_model_type == 'bert_embeddings' or self.language_model_type == 'mixed_bert_embeddings':
            batched_x['word_ids'] = pad_sequence(batched_x['word_ids'], batch_first=True, padding_value=self.padding_token_id)
            batched_x['attention_mask'] = pad_sequence(batched_x['attention_mask'], batch_first=True, padding_value=0)
        else:
            raise ValueError('Unsupported value for input_representation: {}'.format(self.language_model_type))

        batched_x['sequence_lengths'] = torch.as_tensor(batched_x['sequence_lengths'])

        batched_x['subword_indices'] = pad_sequence(
            batched_x['subword_indices'],
            batch_first=True,
            padding_value=max_sequence_length - 1)

        batched_y['predicates'] = pad_sequence(
            batched_y['predicates'],
            batch_first=True,
            padding_value=self.padding_label_id)

        batched_y['senses'] = torch.as_tensor(batched_y['senses'])
        batched_y['rolesets'] = torch.as_tensor(batched_y['rolesets'])
        batched_y['roles'] = ConllDataModule._pad_bidimensional_sequences(
            batched_y['roles'],
            sequence_length=max_sequence_length - 2,
            padding_value=self.padding_label_id)

        return batched_x, batched_y
    
    def save(self, dir, config_name='config.json'):
        config = {}
        config['language_model_type'] = self.language_model_type
        config['language_model_name'] = self.language_model_name
        config['unknown_token'] = self.unknown_token
        config['unknown_sense_id'] = self.unknown_sense_id
        config['unknown_role_id'] = self.unknown_role_id
        config['padding_label_id'] = self.padding_label_id
        config['num_senses'] = self.num_senses
        config['num_roles'] = self.num_roles
        config['sense2id'] = self.sense2id
        config['role2id'] = self.role2id

        config_path = os.path.join(dir, config_name)
        with open(config_path, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    @staticmethod
    def load_from_config(config_path):
        with open(config_path) as f:
            config = json.load(f)

        data_module = ConllDataModule()
        data_module.language_model_type = config['language_model_type']
        data_module.language_model_name = config['language_model_name']
        data_module.unknown_token = config['unknown_token']
        data_module.unknown_sense_id = config['unknown_sense_id']
        data_module.unknown_role_id = config['unknown_role_id']
        data_module.padding_label_id = config['padding_label_id']
        data_module.num_senses = config['num_senses']
        data_module.num_roles = config['num_roles']
        data_module.sense2id = config['sense2id']
        data_module.role2id = config['role2id']

        data_module.id2sense = {id: predicate for predicate, id in data_module.sense2id.items()}
        data_module.id2role = {id: role for role, id in data_module.role2id.items()}

        data_module._setup_tokenizer()

        return data_module

    @staticmethod
    def _pad_bidimensional_sequences(sequences, sequence_length, padding_value=0):
        padded_sequences = torch.full((len(sequences), sequence_length, sequence_length), padding_value, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            for j, subsequence in enumerate(sequence):
                padded_sequences[i][j][:len(subsequence)] = torch.as_tensor(subsequence, dtype=torch.long)
        return padded_sequences

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--dev_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--inventory', type=str, default='propbank')
        parser.add_argument('--span_based', action='store_true')
        return parser