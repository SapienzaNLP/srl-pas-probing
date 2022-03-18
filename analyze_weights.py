from argparse import ArgumentParser

import torch
import torch.nn as nn

import plotly.graph_objects as go

from multilingual_srl.models.probers.propbank_structure_prober import PropbankStructureProber
from multilingual_srl.models.probers.verbatlas_structure_prober import VerbatlasStructureProber

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--inventory', type=str, default='propbank')

    # Store the arguments in hparams.
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.inventory == 'propbank':
        model = PropbankStructureProber.load_from_checkpoint(args.checkpoint)
    elif args.invetory == 'verbatlas':
        model = VerbatlasStructureProber.load_from_checkpoint(args.checkpoint)

    model.to(device)
    model.eval()

    num_layers = model.word_encoder.word_embedding_layer.num_hidden_layers

    sense_weights = model.word_encoder.word_embedding_layer.weights[:num_layers]
    sense_weights = 100. * nn.functional.softmax(torch.cat([w for w in sense_weights]), dim=0)
    sense_weights = sense_weights.tolist()
    print('Sense weights:')
    print(sense_weights)
    print()

    pas_weights = model.word_encoder.word_embedding_layer.weights[num_layers:]
    pas_weights = 100. * nn.functional.softmax(torch.cat([w for w in pas_weights]), dim=0)
    pas_weights = pas_weights.tolist()
    print('Roleset weights:')
    print(pas_weights)
    print()


    x = list(range(1, len(sense_weights) + 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,
                    y=sense_weights,
                    name='Sense',
                    marker_color='rgb(109, 55, 83)'
                    ))
    fig.add_trace(go.Bar(x=x,
                    y=pas_weights,
                    name='Roleset',
                    marker_color='rgb(255, 26, 118)'
                    ))

    fig.update_layout(
        xaxis=dict(
            title='PLM layers',
            titlefont_size=18,
            tickfont_size=16,
        ),
        yaxis=dict(
            title='Relative importance (%)',
            titlefont_size=18,
            tickfont_size=16,
        ),
        legend=dict(
            x=0.05,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            font_size=18,
        ),
        barmode='group',
        bargap=0.5, # gap between bars of adjacent location coordinates.
        bargroupgap=0.0 # gap between bars of the same location coordinate.
    )

    fig.write_image(args.output)
