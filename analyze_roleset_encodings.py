from argparse import ArgumentParser
from collections import Counter

import numpy as np

import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale, normalize

import plotly.express as px
import pandas as pd

from multilingual_srl.models.probers.propbank_structure_prober import PropbankStructureProber
from multilingual_srl.models.probers.verbatlas_structure_prober import VerbatlasStructureProber
from multilingual_srl.data.conll_data_module import ConllDataModule

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--datamodule', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    # Add data-specific args.
    parser = ConllDataModule.add_data_specific_args(parser)

    # Store the arguments in hparams.
    args = parser.parse_args()

    data_module = ConllDataModule.load_from_config(args.datamodule)
    data_module.train_path = args.train_path
    data_module.dev_path = args.dev_path
    data_module.batch_size = args.batch_size
    data_module.num_workers = args.num_workers
    data_module.prepare_data()
    data_module.setup('fit')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.inventory == 'propbank':
        model = PropbankStructureProber.load_from_checkpoint(args.checkpoint)
    elif args.inventory == 'verbatlas':
        model = VerbatlasStructureProber.load_from_checkpoint(args.checkpoint)

    model.to(device)
    model.eval()

    gold_senses = []
    gold_rolesets = []
    pred_rolesets = []
    encodings = []

    with torch.no_grad():
        for x, batch_gold in data_module.train_dataloader():
            x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
            y = model(x, return_encodings=True)

            batch_gold_senses = [data_module.id2sense[s] for s in batch_gold['senses'].tolist()]
            batch_gold_senses = [s for s in batch_gold_senses if s != '_']
            gold_senses.extend(batch_gold_senses)

            batch_gold_rolesets = [[data_module.id2role[i] for i, r in enumerate(roleset) if r == 1.0] for roleset in batch_gold['rolesets'].tolist()]
            gold_rolesets.extend(batch_gold_rolesets)

            batch_pred_rolesets = [[data_module.id2role[i] for i, r in enumerate(roleset) if r > 0.5] for roleset in torch.sigmoid(y['rolesets']).tolist()]
            pred_rolesets.extend(batch_pred_rolesets)

            encodings.extend(y['roleset_encodings'].tolist())
    
    filtered_encodings = []
    filtered_senses = []
    filtered_rolesets = []
    # filter_lemmas = ['end', 'start', 'begin', 'finish']
    filter_lemmas = ['close']
    filter_roles = ['AM-TMP', 'AM-EXT', 'AM-MNR']
    for sense, roleset, encoding in zip(gold_senses, gold_rolesets, encodings):
        lemma = sense.split('-')[0]
        roleset = sorted([r for r in roleset if r in filter_roles])
        if lemma in filter_lemmas and roleset and len(roleset) < 3:
            filtered_senses.append(sense)
            filtered_rolesets.append('/'.join(sorted(roleset, reverse=False)))
            filtered_encodings.append(encoding)

    filtered_encodings = np.array(filtered_encodings)
    filtered_encodings = scale(filtered_encodings)
    filtered_encodings = normalize(filtered_encodings)

    tsne = TSNE(n_iter=5_000, random_state=313, learning_rate='auto', init='pca', early_exaggeration=10.0, perplexity=20.0, verbose=1)
    projections = tsne.fit_transform(filtered_encodings)

    df = pd.DataFrame()
    df['senses'] = filtered_senses
    df['rolesets'] = filtered_rolesets
    df['tsne_0'] = projections[:, 0]
    df['tsne_1'] = projections[:, 1]
    fig = px.scatter(df, x='tsne_0', y='tsne_1', color='rolesets', hover_data=['senses', 'rolesets'])

    fig.show()

    fig.write_image(args.output)
    
