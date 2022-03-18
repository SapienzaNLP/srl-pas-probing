from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from multilingual_srl.layers.word_encoder import WordEncoder
from multilingual_srl.layers.sequence_encoder import SequenceEncoder
from multilingual_srl.layers.state_encoder import StateEncoder


class SrlModel(pl.LightningModule):

    def __init__(
        self,
        num_senses,
        num_roles,

        language_model_type='mixed_bert_embeddings',
        language_model_name='bert-base-multilingual-cased',
        language_model_fine_tuning=False,
        language_model_random_initialization=False,
        use_roleset_encodings=False,

        word_encoding_size=512,
        word_encoding_activation='swish',
        word_encoding_dropout=0.1,

        role_encoding_size=256,
        role_encoding_activation='swish',
        role_encoding_dropout=0.1,

        predicate_timestep_encoding_size=512,
        predicate_timestep_encoding_activation='swish',
        predicate_timestep_encoding_dropout=0.1,

        roleset_timestep_encoding_size=256,
        roleset_timestep_encoding_activation='swish',
        roleset_timestep_encoding_dropout=0.1,

        argument_timestep_encoding_size=512,
        argument_timestep_encoding_activation='swish',
        argument_timestep_encoding_dropout=0.1,
    
        word_sequence_encoder_type='lstm',
        word_sequence_encoder_hidden_size=512,
        word_sequence_encoder_layers=1,
        word_sequence_encoder_dropout=0.1,

        argument_sequence_encoder_type='lstm',
        argument_sequence_encoder_hidden_size=512,
        argument_sequence_encoder_layers=1,
        argument_sequence_encoder_dropout=0.1,

        learning_rate=1e-3,
        min_learning_rate=5e-5,
        weight_decay=1e-4,
        language_model_learning_rate=5e-5,
        language_model_min_learning_rate=1e-8,
        language_model_weight_decay=1e-4,

        padding_label_id=-1,
    ):
        super(SrlModel, self).__init__()

        self.num_roles = num_roles
        self.num_senses = num_senses
        self.padding_label_id = padding_label_id

        self.language_model_type = language_model_type
        self.language_model_name = language_model_name
        self.language_model_fine_tuning = language_model_fine_tuning
        self.langauge_model_random_initialization = language_model_random_initialization
        self.use_roleset_encodings = use_roleset_encodings
        
        self.word_encoding_size = word_encoding_size
        self.word_encoding_activation = word_encoding_activation
        self.word_encoding_dropout = word_encoding_dropout

        self.role_encoding_size = role_encoding_size
        self.role_encoding_activation = role_encoding_activation
        self.role_encoding_dropout = role_encoding_dropout

        self.predicate_timestep_encoding_size = predicate_timestep_encoding_size
        self.predicate_timestep_encoding_activation = predicate_timestep_encoding_activation
        self.predicate_timestep_encoding_dropout = predicate_timestep_encoding_dropout

        self.roleset_timestep_encoding_size = roleset_timestep_encoding_size
        self.roleset_timestep_encoding_activation = roleset_timestep_encoding_activation
        self.roleset_timestep_encoding_dropout = roleset_timestep_encoding_dropout

        self.argument_timestep_encoding_size = argument_timestep_encoding_size
        self.argument_timestep_encoding_activation = argument_timestep_encoding_activation
        self.argument_timestep_encoding_dropout = argument_timestep_encoding_dropout

        self.word_sequence_encoder_type = word_sequence_encoder_type
        self.word_sequence_encoder_hidden_size = word_sequence_encoder_hidden_size
        self.word_sequence_encoder_layers = word_sequence_encoder_layers
        self.word_sequence_encoder_dropout = word_sequence_encoder_dropout

        self.argument_sequence_encoder_type = argument_sequence_encoder_type
        self.argument_sequence_encoder_hidden_size = argument_sequence_encoder_hidden_size
        self.argument_sequence_encoder_layers = argument_sequence_encoder_layers
        self.argument_sequence_encoder_dropout = argument_sequence_encoder_dropout

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
            output_size=self.word_encoding_size,
            activation_type=self.word_encoding_activation,
            dropout_rate=self.word_encoding_dropout,
            num_outputs=3)
        word_embedding_size = self.word_encoder.output_size

        self.sequence_encoder = SequenceEncoder(
            encoder_type=self.word_sequence_encoder_type,
            input_size=word_embedding_size,
            hidden_size=self.word_sequence_encoder_hidden_size,
            num_layers=self.word_sequence_encoder_layers,
            dropout=self.word_sequence_encoder_dropout)
        word_timestep_size = self.sequence_encoder.output_size

        self.predicate_timestep_encoder = StateEncoder(
            input_size=word_embedding_size,
            state_size=self.predicate_timestep_encoding_size,
            activation=self.predicate_timestep_encoding_activation,
            dropout_rate=self.predicate_timestep_encoding_dropout)

        self.roleset_timestep_encoder = StateEncoder(
            input_size=word_embedding_size,
            state_size=self.roleset_timestep_encoding_size,
            activation=self.roleset_timestep_encoding_activation,
            dropout_rate=self.roleset_timestep_encoding_dropout)

        self.argument_timestep_encoder = StateEncoder(
            input_size=word_timestep_size,
            state_size=self.argument_timestep_encoding_size,
            activation=self.argument_timestep_encoding_activation,
            dropout_rate=self.argument_timestep_encoding_dropout)

        argument_sequence_encoder_input_size = self.predicate_timestep_encoding_size + self.argument_timestep_encoding_size
        if self.use_roleset_encodings:
            argument_sequence_encoder_input_size += self.roleset_timestep_encoding_size
        
        self.argument_sequence_encoder = SequenceEncoder(
            encoder_type=self.argument_sequence_encoder_type,
            input_size=argument_sequence_encoder_input_size,
            hidden_size=self.argument_sequence_encoder_hidden_size,
            num_layers=self.argument_sequence_encoder_layers,
            dropout=self.argument_sequence_encoder_dropout)
        predicate_argument_timestep_size = self.argument_sequence_encoder.output_size

        self.argument_encoder = StateEncoder(
            input_size=predicate_argument_timestep_size,
            state_size=self.role_encoding_size,
            activation=self.role_encoding_activation,
            dropout_rate=self.role_encoding_dropout)

        self.predicate_scorer = nn.Linear(word_embedding_size, 2)
        self.sense_scorer = nn.Linear(word_embedding_size, self.num_senses)
        self.roleset_scorer = nn.Linear(word_embedding_size, self.num_roles)
        self.role_scorer = nn.Linear(self.role_encoding_size, self.num_roles)

    def forward(self, x):
        word_ids = x['word_ids']
        attention_mask = x['attention_mask']
        subword_indices = x['subword_indices']
        sequence_lengths = x['sequence_lengths']
        predicate_indices = x['predicate_indices']

        word_encodings, predicate_encodings, roleset_encodings = self.word_encoder(
            word_ids,
            attention_mask,
            subword_indices
        )

        word_sequence_encodings = self.sequence_encoder(word_encodings, sequence_lengths)

        word_sequence_encodings = word_sequence_encodings[:, 1:-1, :]
        predicate_encodings = predicate_encodings[:, 1:-1, :]
        roleset_encodings = roleset_encodings[:, 1:-1, :]
        
        timesteps = word_sequence_encodings.shape[1]

        predicate_timestep_encodings = self.predicate_timestep_encoder(predicate_encodings)
        predicate_timestep_encodings = predicate_timestep_encodings.unsqueeze(2).expand(-1, -1, timesteps, -1)

        roleset_timestep_encodings = self.roleset_timestep_encoder(roleset_encodings)
        roleset_timestep_encodings = roleset_timestep_encodings.unsqueeze(2).expand(-1, -1, timesteps, -1)

        argument_timestep_encodings = self.argument_timestep_encoder(word_sequence_encodings)
        argument_timestep_encodings = argument_timestep_encodings.unsqueeze(1).expand(-1, timesteps, -1, -1)

        if self.use_roleset_encodings:
            predicate_argument_states = torch.cat(
                (predicate_timestep_encodings, roleset_timestep_encodings, argument_timestep_encodings),
                dim=-1)
        else:
            predicate_argument_states = torch.cat(
                (predicate_timestep_encodings, argument_timestep_encodings),
                dim=-1)
        
        predicate_argument_states = predicate_argument_states[predicate_indices]

        argument_sequence_lengths = sequence_lengths[predicate_indices[0]] - 2
        max_argument_sequence_length = torch.max(argument_sequence_lengths)
        predicate_argument_states = predicate_argument_states[:, :max_argument_sequence_length, :]
        argument_encodings = self.argument_sequence_encoder(predicate_argument_states, argument_sequence_lengths)
        argument_encodings = self.argument_encoder(argument_encodings)

        predicate_scores = self.predicate_scorer(predicate_encodings)

        sense_encodings = predicate_encodings[predicate_indices]
        sense_scores = self.sense_scorer(sense_encodings)

        roleset_encodings = roleset_encodings[predicate_indices]
        roleset_scores = self.roleset_scorer(roleset_encodings)

        role_scores = self.role_scorer(argument_encodings)

        return {
            'predicates': predicate_scores,
            'senses': sense_scores,
            'rolesets': roleset_scores,
            'roles': role_scores,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            },
        }

        return optimizer

    def training_step(self, batch, batch_index):
        step_result = self._shared_step(batch)
        self.log('train_loss', step_result['loss'])
        self.log('train_loss_predicate_classification', step_result['predicate_identification_loss'])
        self.log('train_loss_sense_classification', step_result['sense_classification_loss'])
        self.log('train_loss_role_classification', step_result['argument_classification_loss'])
        if self.use_roleset_encodings:
            self.log('train_loss_roleset_classification', step_result['roleset_classification_loss'])
        return step_result['loss']

    def validation_step(self, batch, batch_index):
        return self._shared_step(batch, compute_metrics=True)

    def test_step(self, batch, batch_index):
        return self._shared_step(batch, compute_metrics=True)

    def _shared_step(self, batch, compute_metrics=False):
        sample, labels = batch
        scores = self(sample)

        labels['roles'] = labels['roles'][sample['predicate_indices']]
        
        predicate_identification_loss = SrlModel._compute_classification_loss(
            scores['predicates'],
            labels['predicates'],
            2,
            ignore_index=self.padding_label_id,
        )
        sense_classification_loss = SrlModel._compute_classification_loss(
            scores['senses'],
            labels['senses'],
            self.num_senses,
            ignore_index=self.padding_label_id,
        )
        argument_classification_loss = SrlModel._compute_classification_loss(
            scores['roles'],
            labels['roles'],
            self.num_roles,
            ignore_index=self.padding_label_id,
        )

        if self.use_roleset_encodings:
            roleset_classification_loss = SrlModel._compute_binary_classification_loss(
                scores['rolesets'],
                labels['rolesets'],
            )

        loss = predicate_identification_loss + sense_classification_loss + argument_classification_loss
        if self.use_roleset_encodings:
            loss += roleset_classification_loss

        metrics = self.compute_step_metrics(scores, labels) if compute_metrics else {}

        if torch.isnan(loss) or not torch.isfinite(loss):
            self.print('Loss:', loss)
            self.print('Predicate identification loss:', predicate_identification_loss)
            self.print('Predicate disambiguation loss:', sense_classification_loss)
            self.print('Argument classification loss:', argument_classification_loss)
            if self.use_roleset_encodings:
                self.print('Roleset classification loss:', roleset_classification_loss)
            raise ValueError('NaN loss!')

        return_values = {
            'loss': loss,
            'predicate_identification_loss': predicate_identification_loss,
            'sense_classification_loss': sense_classification_loss,
            'argument_classification_loss': argument_classification_loss,
            'metrics': metrics,
        }

        if self.use_roleset_encodings:
            return_values['roleset_classification_loss'] = roleset_classification_loss

        return return_values

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')

    def _shared_epoch_end(self, outputs, stage):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        predicate_identification_loss = torch.stack([x['predicate_identification_loss'] for x in outputs]).mean()
        sense_classification_loss = torch.stack([x['sense_classification_loss'] for x in outputs]).mean()
        argument_classification_loss = torch.stack([x['argument_classification_loss'] for x in outputs]).mean()
        if self.use_roleset_encodings:
            roleset_classification_loss = torch.stack([x['roleset_classification_loss'] for x in outputs]).mean()
        
        metrics = SrlModel._compute_epoch_metrics(outputs, compute_roleset_metrics=self.use_roleset_encodings)

        logs = {
            f'{stage}_loss': avg_loss,
            f'{stage}_loss_predicate_identification': predicate_identification_loss,
            f'{stage}_loss_sense_classification': sense_classification_loss,
            f'{stage}_loss_argument_classification': argument_classification_loss,
            f'{stage}_predicate_precision': metrics['predicates']['precision'],
            f'{stage}_predicate_recall': metrics['predicates']['recall'],
            f'{stage}_predicate_f1': metrics['predicates']['f1'],
            f'{stage}_sense_accuracy': metrics['senses']['sense_accuracy'],
            f'{stage}_role_precision': metrics['roles']['precision'],
            f'{stage}_role_recall': metrics['roles']['recall'],
            f'{stage}_role_f1': metrics['roles']['f1'],
            f'{stage}_overall_precision': metrics['overall']['precision'],
            f'{stage}_overall_recall': metrics['overall']['recall'],
            f'{stage}_overall_f1': metrics['overall']['f1'],
        }

        if self.use_roleset_encodings:
            logs.update({
                f'{stage}_loss_roleset_classification': roleset_classification_loss,
                f'{stage}_roleset_accuracy': metrics['rolesets']['roleset_accuracy'],
                f'{stage}_roleset_precision': metrics['rolesets']['roleset_precision'],
                f'{stage}_roleset_recall': metrics['rolesets']['roleset_recall'],
                f'{stage}_roleset_f1': metrics['rolesets']['roleset_f1'],
                f'{stage}_core_roleset_accuracy': metrics['rolesets']['core_roleset_accuracy'],
                f'{stage}_core_roleset_precision': metrics['rolesets']['core_roleset_precision'],
                f'{stage}_core_roleset_recall': metrics['rolesets']['core_roleset_recall'],
                f'{stage}_core_roleset_f1': metrics['rolesets']['core_roleset_f1'],
            })

        self.log_dict(logs)

    @staticmethod
    def _compute_classification_loss(scores, labels, num_classes, ignore_index=-1):
        classification_loss = F.cross_entropy(
            scores.view(-1, num_classes),
            labels.view(-1),
            reduction='sum',
            ignore_index=ignore_index)

        return classification_loss

    @staticmethod
    def _compute_binary_classification_loss(scores, labels):
        classification_loss = F.binary_cross_entropy_with_logits(
            scores,
            labels,
            reduction='sum')

        return classification_loss

    def compute_step_metrics(self, scores, labels):
        predicates_g = labels['predicates']
        predicates_p = torch.argmax(scores['predicates'], dim=-1)
        predicate_tp = (predicates_p[torch.logical_and(predicates_g >= 0, predicates_p == 1)] == predicates_g[torch.logical_and(predicates_g >= 0, predicates_p == 1)]).sum()
        predicate_fp = (predicates_p[torch.logical_and(predicates_g >= 0, predicates_p == 1)] != predicates_g[torch.logical_and(predicates_g >= 0, predicates_p == 1)]).sum()
        predicate_fn = (predicates_p[predicates_g == 1] != predicates_g[predicates_g == 1]).sum()

        senses_g = labels['senses']
        senses_p = torch.argmax(scores['senses'], dim=-1)
        correct_senses = (senses_g == senses_p).sum()
        total_senses = torch.as_tensor(senses_g.shape[-1])

        roles_g = labels['roles']
        roles_p = torch.argmax(scores['roles'], dim=-1)
        role_tp = (roles_p[torch.logical_and(roles_g >= 0, roles_p >= 1)] == roles_g[torch.logical_and(roles_g >= 0, roles_p >= 1)]).sum()
        role_fp = (roles_p[torch.logical_and(roles_g >= 0, roles_p >= 1)] != roles_g[torch.logical_and(roles_g >= 0, roles_p >= 1)]).sum()
        role_fn = (roles_p[roles_g >= 1] != roles_g[roles_g >= 1]).sum()

        if self.use_roleset_encodings:
            rolesets_g = labels['rolesets'].view(-1)
            rolesets_p = (torch.sigmoid(scores['rolesets']) > 0.5).to(torch.float).view(-1)
            correct_rolesets = (rolesets_g == rolesets_p).sum()
            total_rolesets = torch.as_tensor(rolesets_g.shape[-1])
            tp_rolesets = (rolesets_p[rolesets_p == 1.0] == rolesets_g[rolesets_p == 1.0]).sum()
            fp_rolesets = (rolesets_p[rolesets_p == 1.0] != rolesets_g[rolesets_p == 1.0]).sum()
            fn_rolesets = (rolesets_p[rolesets_g == 1.0] != rolesets_g[rolesets_g == 1.0]).sum()

            core_rolesets_g = labels['rolesets'][:, 2:8].reshape(-1)
            core_rolesets_p = (torch.sigmoid(scores['rolesets'][:, 2:8]) > 0.5).to(torch.float).reshape(-1)
            correct_core_rolesets = (core_rolesets_g == core_rolesets_p).sum()
            total_core_rolesets = torch.as_tensor(core_rolesets_g.shape[-1])
            tp_core_rolesets = (core_rolesets_p[core_rolesets_p == 1.0] == core_rolesets_g[core_rolesets_p == 1.0]).sum()
            fp_core_rolesets = (core_rolesets_p[core_rolesets_p == 1.0] != core_rolesets_g[core_rolesets_p == 1.0]).sum()
            fn_core_rolesets = (core_rolesets_p[core_rolesets_g == 1.0] != core_rolesets_g[core_rolesets_g == 1.0]).sum()

        return_values = {
            'predicate_tp': predicate_tp,
            'predicate_fp': predicate_fp,
            'predicate_fn': predicate_fn,

            'correct_senses': correct_senses,
            'total_senses': total_senses,

            'role_tp': role_tp,
            'role_fp': role_fp,
            'role_fn': role_fn,
        }

        if self.use_roleset_encodings:
            return_values.update({
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
            })

        return return_values

    @staticmethod
    def _compute_epoch_metrics(outputs, compute_roleset_metrics=False):
        predicate_tp = torch.stack([o['metrics']['predicate_tp'] for o in outputs]).sum()
        predicate_fp = torch.stack([o['metrics']['predicate_fp'] for o in outputs]).sum()
        predicate_fn = torch.stack([o['metrics']['predicate_fn'] for o in outputs]).sum()

        predicate_precision = torch.true_divide(predicate_tp, (predicate_tp + predicate_fp)) if predicate_tp + predicate_fp > 0 else torch.as_tensor(0)
        predicate_recall = torch.true_divide(predicate_tp, (predicate_tp + predicate_fn)) if predicate_tp + predicate_fn > 0 else torch.as_tensor(0)
        predicate_f1 = 2 * torch.true_divide(predicate_precision * predicate_recall, predicate_precision + predicate_recall) if predicate_precision + predicate_recall > 0 else torch.as_tensor(0)

        correct_senses = torch.stack([o['metrics']['correct_senses'] for o in outputs]).sum()
        total_senses = torch.stack([o['metrics']['total_senses'] for o in outputs]).sum()
        sense_accuracy = torch.true_divide(correct_senses, total_senses)

        role_tp = torch.stack([o['metrics']['role_tp'] for o in outputs]).sum()
        role_fp = torch.stack([o['metrics']['role_fp'] for o in outputs]).sum()
        role_fn = torch.stack([o['metrics']['role_fn'] for o in outputs]).sum()

        role_precision = torch.true_divide(role_tp, (role_tp + role_fp)) if role_tp + role_fp > 0 else torch.as_tensor(0)
        role_recall = torch.true_divide(role_tp, (role_tp + role_fn)) if role_tp + role_fn > 0 else torch.as_tensor(0)
        role_f1 = 2 * torch.true_divide(role_precision * role_recall, role_precision + role_recall) if role_precision + role_recall > 0 else torch.as_tensor(0)

        overall_tp = role_tp + correct_senses
        overall_fp = role_fp + total_senses - correct_senses
        overall_fn = role_fn + total_senses - correct_senses
        overall_precision = torch.true_divide(overall_tp, (overall_tp + overall_fp)) if overall_tp + overall_fp > 0 else torch.as_tensor(0)
        overall_recall = torch.true_divide(overall_tp, (overall_tp + overall_fn)) if overall_tp + overall_fn > 0 else torch.as_tensor(0)
        overall_f1 = 2 * torch.true_divide(overall_precision * overall_recall, overall_precision + overall_recall) if overall_precision + overall_recall > 0 else torch.as_tensor(0)

        if compute_roleset_metrics:
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

        return_values = {
            'predicates': {
                '_tp': predicate_tp,
                '_fp': predicate_fp,
                '_fn': predicate_fn,
                'precision': predicate_precision,
                'recall': predicate_recall,
                'f1': predicate_f1,
            },
            'senses': {
                '_correct_senses': correct_senses,
                '_total_senses': total_senses,
                'sense_accuracy': sense_accuracy,
            },
            'roles': {
                '_tp': role_tp,
                '_fp': role_fp,
                '_fn': role_fn,
                'precision': role_precision,
                'recall': role_recall,
                'f1': role_f1,
            },
            'overall': {
                '_tp': overall_tp,
                '_fp': overall_fp,
                '_fn': overall_fn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
            },
        }

        if compute_roleset_metrics:
            return_values.update({
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
            })

        return return_values

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--language_model_type', type=str, default='mixed_bert_embeddings')
        parser.add_argument('--language_model_name', type=str, default='bert-base-cased')
        parser.add_argument('--language_model_fine_tuning', action='store_true')
        parser.add_argument('--language_model_random_initialization', action='store_true')
        parser.add_argument('--use_roleset_encodings', action='store_true')
        parser.add_argument('--word_encoding_size', type=int, default=512)
        parser.add_argument('--word_encoding_activation', type=str, default='swish')
        parser.add_argument('--word_encoding_dropout', type=float, default=0.2)

        parser.add_argument('--predicate_timestep_encoding_size', type=int, default=512)
        parser.add_argument('--predicate_timestep_encoding_activation', type=str, default='swish')
        parser.add_argument('--predicate_timestep_encoding_dropout', type=float, default=0.1)

        parser.add_argument('--roleset_timestep_encoding_size', type=int, default=256)
        parser.add_argument('--roleset_timestep_encoding_activation', type=str, default='swish')
        parser.add_argument('--roleset_timestep_encoding_dropout', type=float, default=0.1)

        parser.add_argument('--argument_timestep_encoding_size', type=int, default=512)
        parser.add_argument('--argument_timestep_encoding_activation', type=str, default='swish')
        parser.add_argument('--argument_timestep_encoding_dropout', type=float, default=0.1)

        parser.add_argument('--word_sequence_encoder_type', type=str, default='connected_lstm')
        parser.add_argument('--word_sequence_encoder_hidden_size', type=int, default=512) 
        parser.add_argument('--word_sequence_encoder_layers', type=int, default=2)
        parser.add_argument('--word_sequence_encoder_dropout', type=float, default=0.2)

        parser.add_argument('--argument_sequence_encoder_type', type=str, default='connected_lstm')
        parser.add_argument('--argument_sequence_encoder_hidden_size', type=int, default=512)
        parser.add_argument('--argument_sequence_encoder_layers', type=int, default=1)
        parser.add_argument('--argument_sequence_encoder_dropout', type=float, default=0.2)

        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--min_learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--language_model_learning_rate', type=float, default=1e-5)
        parser.add_argument('--language_model_min_learning_rate', type=float, default=1e-6)
        parser.add_argument('--language_model_weight_decay', type=float, default=1e-2)
        return parser
