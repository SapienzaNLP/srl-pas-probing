from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from multilingual_srl.layers.word_encoder import WordEncoder


class VerbatlasStructureProber(pl.LightningModule):

    def __init__(
        self,
        num_senses,
        num_roles,

        language_model_type='mixed_bert_embeddings',
        language_model_name='bert-base-multilingual-cased',
        language_model_fine_tuning=False,
        language_model_random_initialization=False,
        return_static_embeddings=False,

        word_encoding_size=512,
        word_encoding_activation='swish',
        word_encoding_dropout=0.1,

        learning_rate=1e-3,
        min_learning_rate=1e-4,
        weight_decay=1e-2,
        language_model_learning_rate=1e-5,
        language_model_min_learning_rate=1e-6,
        language_model_weight_decay=1e-2,

        padding_label_id=-1,
    ):
        super().__init__()

        self.num_roles = num_roles
        self.num_senses = num_senses
        self.padding_label_id = padding_label_id

        self.language_model_type = language_model_type
        self.language_model_name = language_model_name
        self.language_model_fine_tuning = language_model_fine_tuning
        self.word_encoding_size = word_encoding_size
        self.word_encoding_activation = word_encoding_activation
        self.word_encoding_dropout = word_encoding_dropout

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.language_model_learning_rate = language_model_learning_rate
        self.language_model_min_learning_rate = language_model_min_learning_rate
        self.language_model_weight_decay = language_model_weight_decay

        self.save_hyperparameters()

        self.word_encoder = WordEncoder(
            language_model_type=self.language_model_type,
            language_model_name=self.language_model_name,
            language_model_fine_tuning=self.language_model_fine_tuning,
            language_model_random_initialization=language_model_random_initialization,
            return_static_embeddings=return_static_embeddings,
            output_size=self.word_encoding_size,
            activation_type=self.word_encoding_activation,
            dropout_rate=self.word_encoding_dropout,
            num_outputs=2)
        word_embedding_size = self.word_encoder.output_size

        self.sense_scorer = nn.Linear(word_embedding_size, self.num_senses)
        self.roleset_scorer = nn.Linear(word_embedding_size, self.num_roles)

    def forward(self, x, return_encodings=False):
        word_ids = x['word_ids']
        attention_mask = x['attention_mask']
        subword_indices = x['subword_indices']
        predicate_indices = x['predicate_indices']

        sense_encodings, roleset_encodings = self.word_encoder(
            word_ids,
            attention_mask,
            subword_indices
        )
        sense_encodings = sense_encodings[:, 1:-1, :]
        roleset_encodings = roleset_encodings[:, 1:-1, :]

        sense_encodings = sense_encodings[predicate_indices]
        sense_scores = self.sense_scorer(sense_encodings)

        roleset_encodings = roleset_encodings[predicate_indices]
        roleset_scores = self.roleset_scorer(roleset_encodings)

        if not return_encodings:
            outputs = {
                'senses': sense_scores,
                'rolesets': roleset_scores,
            }
        else:
            outputs = {
                'senses': sense_scores,
                'rolesets': roleset_scores,
                'sense_encodings': sense_encodings,
                'roleset_encodings': roleset_encodings,
            }
        
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        return optimizer

    def training_step(self, batch, batch_index):
        step_result = self._shared_step(batch)
        self.log('train_loss', step_result['loss'])
        self.log('train_loss_sense_classification', step_result['sense_classification_loss'])
        self.log('train_loss_roleset_classification', step_result['roleset_classification_loss'])
        return step_result['loss']

    def validation_step(self, batch, batch_index):
        return self._shared_step(batch, compute_metrics=True)

    def test_step(self, batch, batch_index):
        return self._shared_step(batch, compute_metrics=True)

    def _shared_step(self, batch, compute_metrics=False):
        sample, labels = batch
        scores = self(sample)
        
        sense_classification_loss = VerbatlasStructureProber._compute_classification_loss(
            scores['senses'],
            labels['senses'],
            self.num_senses,
            ignore_index=self.padding_label_id,
        )
        roleset_classification_loss = VerbatlasStructureProber._compute_binary_classification_loss(
            scores['rolesets'],
            labels['rolesets'],
        )

        loss = sense_classification_loss + roleset_classification_loss
        metrics = self.compute_step_metrics(scores, labels) if compute_metrics else {}

        if torch.isnan(loss) or not torch.isfinite(loss):
            self.print('Loss:', loss)
            self.print('Predicate disambiguation loss:', sense_classification_loss)
            self.print('roleset classification loss:', roleset_classification_loss)
            raise ValueError('NaN loss!')

        return {
            'loss': loss,
            'sense_classification_loss': sense_classification_loss,
            'roleset_classification_loss': roleset_classification_loss,
            'metrics': metrics,
        }

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')

    def _shared_epoch_end(self, outputs, stage):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        sense_classification_loss = torch.stack([x['sense_classification_loss'] for x in outputs]).mean()
        roleset_classification_loss = torch.stack([x['roleset_classification_loss'] for x in outputs]).mean()
        metrics = VerbatlasStructureProber._compute_epoch_metrics(outputs)

        logs = {
            f'{stage}_loss': avg_loss,
            f'{stage}_loss_sense_classification': sense_classification_loss,
            f'{stage}_loss_roleset_classification': roleset_classification_loss,
            f'{stage}_sense_accuracy': metrics['senses']['sense_accuracy'],
            f'{stage}_roleset_accuracy': metrics['rolesets']['roleset_accuracy'],
            f'{stage}_roleset_precision': metrics['rolesets']['roleset_precision'],
            f'{stage}_roleset_recall': metrics['rolesets']['roleset_recall'],
            f'{stage}_roleset_f1': metrics['rolesets']['roleset_f1'],
            f'{stage}_core_roleset_accuracy': metrics['rolesets']['core_roleset_accuracy'],
            f'{stage}_core_roleset_precision': metrics['rolesets']['core_roleset_precision'],
            f'{stage}_core_roleset_recall': metrics['rolesets']['core_roleset_recall'],
            f'{stage}_core_roleset_f1': metrics['rolesets']['core_roleset_f1'],
        }

        self.log_dict(logs)

    @staticmethod
    def _compute_classification_loss(scores, labels, num_classes, ignore_index=-1):
        classification_loss = F.cross_entropy(
            scores.view(-1, num_classes),
            labels.view(-1),
            ignore_index=ignore_index)

        return classification_loss

    @staticmethod
    def _compute_binary_classification_loss(scores, labels):
        classification_loss = F.binary_cross_entropy_with_logits(
            scores,
            labels)

        return classification_loss

    
    def compute_step_metrics(self, scores, labels):
        senses_g = labels['senses']
        senses_p = torch.argmax(scores['senses'], dim=-1)
        correct_senses = (senses_g == senses_p).sum()
        total_senses = torch.as_tensor(senses_g.shape[-1])

        rolesets_g = labels['rolesets'].view(-1)
        rolesets_p = (torch.sigmoid(scores['rolesets']) > 0.5).to(torch.float).view(-1)
        correct_rolesets = (rolesets_g == rolesets_p).sum()
        total_rolesets = torch.as_tensor(rolesets_g.shape[-1])
        tp_rolesets = (rolesets_p[rolesets_p == 1.0] == rolesets_g[rolesets_p == 1.0]).sum()
        fp_rolesets = (rolesets_p[rolesets_p == 1.0] != rolesets_g[rolesets_p == 1.0]).sum()
        fn_rolesets = (rolesets_p[rolesets_g == 1.0] != rolesets_g[rolesets_g == 1.0]).sum()

        core_rolesets_g = labels['rolesets'][:, 2:29].reshape(-1)
        core_rolesets_p = (torch.sigmoid(scores['rolesets'][:, 2:29]) > 0.5).to(torch.float).reshape(-1)
        correct_core_rolesets = (core_rolesets_g == core_rolesets_p).sum()
        total_core_rolesets = torch.as_tensor(core_rolesets_g.shape[-1])
        tp_core_rolesets = (core_rolesets_p[core_rolesets_p == 1.0] == core_rolesets_g[core_rolesets_p == 1.0]).sum()
        fp_core_rolesets = (core_rolesets_p[core_rolesets_p == 1.0] != core_rolesets_g[core_rolesets_p == 1.0]).sum()
        fn_core_rolesets = (core_rolesets_p[core_rolesets_g == 1.0] != core_rolesets_g[core_rolesets_g == 1.0]).sum()

        return {
            'correct_senses': correct_senses,
            'total_senses': total_senses,
            'correct_rolesets': correct_rolesets,
            'total_rolesets': total_rolesets,
            'tp_rolesets': tp_rolesets,
            'fp_rolesets': fp_rolesets,
            'fn_rolesets': fn_rolesets,
            'correct_core_rolesets': correct_core_rolesets,
            'total_core_rolesets': total_core_rolesets,
            'tp_core_rolesets': tp_core_rolesets,
            'fp_core_rolesets': fp_core_rolesets,
            'fn_core_rolesets': fn_core_rolesets,
        }

    @staticmethod
    def _compute_epoch_metrics(outputs):

        correct_senses = torch.stack([o['metrics']['correct_senses'] for o in outputs]).sum()
        total_senses = torch.stack([o['metrics']['total_senses'] for o in outputs]).sum()
        sense_accuracy = torch.true_divide(correct_senses, total_senses)

        correct_rolesets = torch.stack([o['metrics']['correct_rolesets'] for o in outputs]).sum()
        total_rolesets = torch.stack([o['metrics']['total_rolesets'] for o in outputs]).sum()
        roleset_accuracy = torch.true_divide(correct_rolesets, total_rolesets)

        tp_rolesets = torch.stack([o['metrics']['tp_rolesets'] for o in outputs]).sum()
        fp_rolesets = torch.stack([o['metrics']['fp_rolesets'] for o in outputs]).sum()
        fn_rolesets = torch.stack([o['metrics']['fn_rolesets'] for o in outputs]).sum()
        roleset_precision = torch.true_divide(tp_rolesets, tp_rolesets + fp_rolesets)
        roleset_recall = torch.true_divide(tp_rolesets, tp_rolesets + fn_rolesets)
        roleset_f1 = torch.true_divide(2. * roleset_precision * roleset_recall, roleset_precision + roleset_recall)

        correct_core_rolesets = torch.stack([o['metrics']['correct_core_rolesets'] for o in outputs]).sum()
        total_core_rolesets = torch.stack([o['metrics']['total_core_rolesets'] for o in outputs]).sum()
        core_roleset_accuracy = torch.true_divide(correct_core_rolesets, total_core_rolesets)

        tp_core_rolesets = torch.stack([o['metrics']['tp_core_rolesets'] for o in outputs]).sum()
        fp_core_rolesets = torch.stack([o['metrics']['fp_core_rolesets'] for o in outputs]).sum()
        fn_core_rolesets = torch.stack([o['metrics']['fn_core_rolesets'] for o in outputs]).sum()
        core_roleset_precision = torch.true_divide(tp_core_rolesets, tp_core_rolesets + fp_core_rolesets)
        core_roleset_recall = torch.true_divide(tp_core_rolesets, tp_core_rolesets + fn_core_rolesets)
        core_roleset_f1 = torch.true_divide(2. * core_roleset_precision * core_roleset_recall, core_roleset_precision + core_roleset_recall)

        return {
            'senses': {
                '_correct_senses': correct_senses,
                '_total_senses': total_senses,
                'sense_accuracy': sense_accuracy,
            },
            'rolesets': {
                '_correct_rolesets': correct_rolesets,
                '_total_rolesets': total_rolesets,
                '_tp_rolesets': tp_rolesets,
                '_fp_rolesets': fp_rolesets,
                '_fn_rolesets': fn_rolesets,
                '_tp_core_rolesets': tp_core_rolesets,
                '_fp_core_rolesets': fp_core_rolesets,
                '_fn_core_rolesets': fn_core_rolesets,
                'roleset_accuracy': roleset_accuracy,
                'roleset_precision': roleset_precision,
                'roleset_recall': roleset_recall,
                'roleset_f1': roleset_f1,
                'core_roleset_accuracy': core_roleset_accuracy,
                'core_roleset_precision': core_roleset_precision,
                'core_roleset_recall': core_roleset_recall,
                'core_roleset_f1': core_roleset_f1,
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--language_model_type', type=str, default='mixed_bert_embeddings')
        parser.add_argument('--language_model_name', type=str, default='bert-base-multilingual-cased')
        parser.add_argument('--language_model_fine_tuning', default=False, action='store_true')
        parser.add_argument('--language_model_random_initialization', default=False, action='store_true')
        parser.add_argument('--return_static_embeddings', default=False, action='store_true')
        parser.add_argument('--word_encoding_size', type=int, default=512)
        parser.add_argument('--word_encoding_activation', type=str, default='identity')
        parser.add_argument('--word_encoding_dropout', type=float, default=0.2)

        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--min_learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--language_model_learning_rate', type=float, default=1e-5)
        parser.add_argument('--language_model_min_learning_rate', type=float, default=1e-6)
        parser.add_argument('--language_model_weight_decay', type=float, default=1e-2)
        return parser
