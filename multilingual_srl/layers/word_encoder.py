import torch
import torch.nn as nn
import torch_scatter as scatter
from transformers import AutoModel, AutoConfig

from multilingual_srl.layers.swish import Swish


class WordEncoder(nn.Module):

    def __init__(
        self,
        language_model_type='bert_mixed_embeddings',
        language_model_name='bert-base-multilingual-cased',
        language_model_fine_tuning=False,
        language_model_random_initialization=False,
        return_static_embeddings=False,
        output_size=512,
        activation_type='identity',
        dropout_rate=0.1,
        num_outputs=1,
    ):
        super(WordEncoder, self).__init__()

        self.language_model_type = language_model_type
        self.language_model_name = language_model_name
        self.language_model_fine_tuning = language_model_fine_tuning
        self.output_size = output_size
        self.activation = activation_type
        self.dropout_rate = dropout_rate
        self.num_outputs = num_outputs

        if self.language_model_type == 'bert_embeddings':
            self.word_embedding_layer = BertEmbedding(
                model_name=self.language_model_name,
                fine_tune=self.language_model_fine_tuning)
            
            if 'base' in self.language_model_name:
                word_embedding_size = 768
            else:
                word_embedding_size = 1024

        elif self.language_model_type == 'mixed_bert_embeddings':
            self.word_embedding_layer = MixedBertEmbedding(
                model_name=self.language_model_name,
                fine_tune=self.language_model_fine_tuning,
                language_model_random_initialization=language_model_random_initialization,
                return_static_embeddings=return_static_embeddings,
                num_outputs=num_outputs)

            if 'base' in self.language_model_name:
                word_embedding_size = 768
            else:
                word_embedding_size = 1024
        
        else:
            raise NotImplementedError('{} is not implemented.'.format(self.language_model_type))

        self.projection_layer = nn.ModuleList()
        self.dropout_layer = nn.ModuleList()

        for _ in range(self.num_outputs):
            projection_layer = nn.Linear(word_embedding_size, self.output_size)
            dropout_layer = nn.Dropout(self.dropout_rate)
            self.projection_layer.append(projection_layer)
            self.dropout_layer.append(dropout_layer)

        if self.activation == 'identity':
            self.activation_layer = nn.Identity()
        elif self.activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif self.activation == 'swish':
            self.activation_layer = Swish()

    def forward(self, word_ids, attention_mask, subword_indices):
        embeddings = self.word_embedding_layer(word_ids, attention_mask)

        output_encodings = []
        for i in range(self.num_outputs):
            if self.language_model_type == 'bert_embeddings':
                _embeddings = embeddings
            elif self.language_model_type == 'mixed_bert_embeddings':
                _embeddings = embeddings[i]
            
            encodings = self.dropout_layer[i](_embeddings)
            encodings = self.projection_layer[i](encodings)
            encodings = self.activation_layer(encodings)
            encodings = scatter.scatter_mean(encodings, subword_indices, dim=1)
            
            output_encodings.append(encodings)

        return output_encodings


class BertEmbedding(nn.Module):

    def __init__(self, model_name, fine_tune=False):
        super(BertEmbedding, self).__init__()
        self.fine_tune = fine_tune
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        if not fine_tune:
            self.bert.eval()

    def forward(self, word_ids, attention_mask):
        if not self.fine_tune:
            with torch.no_grad():
                word_embeddings = self.bert(
                    input_ids=word_ids,
                    attention_mask=attention_mask)
        else:
            word_embeddings = self.bert(
                input_ids=word_ids,
                attention_mask=attention_mask)

        word_embeddings = 0.25 * sum(word_embeddings[2][-4:])
        return word_embeddings


class MixedBertEmbedding(nn.Module):

    def __init__(self,
        model_name='bert-base-multilingual-cased',
        num_outputs=1,
        language_model_random_initialization=False,
        return_static_embeddings=False,
        fine_tune=False
    ):
        super(MixedBertEmbedding, self).__init__()
        self.return_static_embeddings = return_static_embeddings
        self.fine_tune = fine_tune
        self.num_outputs = num_outputs

        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        if not language_model_random_initialization:
            self.bert = AutoModel.from_pretrained(model_name, config=config)
        else:
            self.bert = AutoModel.from_config(config=config)

        if not fine_tune:
            self.bert.eval()
        
        self.num_hidden_layers = config.num_hidden_layers
        self.gamma = nn.ParameterList()
        self.weights = nn.ParameterList()
        for i in range(num_outputs):
            i = str(i)
            g = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
            w = nn.ParameterList([nn.Parameter(torch.ones([1]), requires_grad=True) for _ in range(self.num_hidden_layers)])
            self.gamma.append(g)
            self.weights.extend(w)

    def forward(self, word_ids, attention_mask):
        if not self.fine_tune:
            with torch.no_grad():
                word_embeddings = self.bert(
                    input_ids=word_ids,
                    attention_mask=attention_mask)
        else:
            word_embeddings = self.bert(
                input_ids=word_ids,
                attention_mask=attention_mask)
        
        mixed_embeddings = []
        for i in range(self.num_outputs):
            if self.return_static_embeddings:
                mixed_embeddings.append(word_embeddings[2][0])
                continue

            _mixed_embeddings = []
            normed_weights = self.weights[i*self.num_hidden_layers: (i+1)*self.num_hidden_layers]
            normed_weights = nn.functional.softmax(torch.cat([w for w in normed_weights]), dim=0)
            normed_weights = torch.split(normed_weights, split_size_or_sections=1)

            for weight, tensor in zip(normed_weights, word_embeddings[2][-self.num_hidden_layers:]):
                _mixed_embeddings.append(weight * tensor)

            mixed_embeddings.append(self.gamma[i] * sum(_mixed_embeddings))
        
        return mixed_embeddings
