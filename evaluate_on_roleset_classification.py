from argparse import ArgumentParser

import torch

from multilingual_srl.data.conll_data_module import ConllDataModule
from multilingual_srl.data.conll_dataset import ConllDataset
from multilingual_srl.models.probers.propbank_structure_prober import PropbankStructureProber
from multilingual_srl.models.probers.verbatlas_structure_prober import VerbatlasStructureProber

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
    data_module.test_path = args.test_path
    data_module.span_based = args.span_based
    data_module.batch_size = args.batch_size
    data_module.num_workers = args.num_workers
    data_module.prepare_data()
    data_module.setup('test')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.inventory == 'propbank':
        model = PropbankStructureProber.load_from_checkpoint(args.checkpoint)
    elif args.invetory == 'verbatlas':
        model = VerbatlasStructureProber.load_from_checkpoint(args.checkpoint)

    model.to(device)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for x, _ in data_module.test_dataloader():
            x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
            y = model(x)
            batch_predictions = data_module.decode(x, y)

            for i, sentence_id in enumerate(x['sentence_ids']):
                predictions[sentence_id] = {
                    'senses': batch_predictions['senses'][i],
                    'rolesets': batch_predictions['rolesets'][i],
                }

    sense_correct = 0
    sense_number_correct = 0
    sense_total = 0

    precision = {}
    recall = {}
    f1 = {}

    with open(args.output, 'w') as f_out:

        for sentence in data_module.test_data:
            sentence_id = sentence['sentence_id']
            sentence_predictions = predictions[sentence_id]
            words = sentence['words']
            words = ' '.join(words)
            f_out.write(f'{sentence_id}\t{words}\n')

            for predicate_index in sentence_predictions['senses']:
                p_sense = sentence_predictions['senses'][predicate_index]
                g_sense = sentence['predicates'][predicate_index]
                
                sense_total += 1
                if p_sense == g_sense:
                    sense_correct += 1
                if p_sense.split('.')[-1] == g_sense.split('.')[-1]:
                    sense_number_correct += 1

                p_roles = sorted(sentence_predictions['rolesets'][predicate_index])
                g_roles = sorted([r for r in sentence['rolesets'][predicate_index] if r != '_'])

                for p_r in p_roles:
                    if p_r not in precision:
                        precision[p_r] = {'n': 0, 'd': 0}
                    precision[p_r]['d'] += 1
                    if p_r in g_roles:
                        precision[p_r]['n'] += 1

                for g_r in g_roles:
                    if g_r not in recall:
                        recall[g_r] = {'n': 0, 'd': 0}
                    recall[g_r]['d'] += 1
                    if g_r in p_roles:
                        recall[g_r]['n'] += 1

                p_roles_str = '\t'.join(p_roles)
                g_roles_str = '\t'.join(g_roles)
                f_out.write(f'Pred:\t{p_sense}\t{p_roles_str}\n')
                f_out.write(f'Gold:\t{g_sense}\t{g_roles_str}\n')
            
            f_out.write('\n')
    
    print()
    print('Sense accuracy: {:0.2f} ({}/{})'.format(100.*sense_correct/sense_total, sense_correct, sense_total))
    print('Sense number accuracy: {:0.2f} ({}/{})'.format(100.*sense_number_correct/sense_total, sense_number_correct, sense_total))
    print()

    print('Precision:')
    precision_values = []
    for r in precision:
        n, d = precision[r]['n'], precision[r]['d']
        v = n / d
        precision_values.append((r, v, n, d))

    precision_values.sort(reverse=True, key=lambda x: x[1])
    num, den = 0, 0
    for r, v, n, d in precision_values:
        num += n
        den += d
        print('  {}{:0.2f} ({}/{})'.format(r.ljust(12), 100. * v, n, d))
    
    micro_precision = num / den if den > 0 else 0.
    print()

    print('Recall:')
    recall_values = []
    for r in recall:
        n, d = recall[r]['n'], recall[r]['d']
        v = n / d
        recall_values.append((r, v, n, d))

    recall_values.sort(reverse=True, key=lambda x: x[1])
    num, den = 0, 0
    for r, v, n, d in recall_values:
        num += n
        den += d
        print('  {}{:0.2f} ({}/{})'.format(r.ljust(12), 100. * v, n, d))

    micro_recall = num / den if den > 0 else 0.

    print('Micro precision: {:0.2f}'.format(100 * micro_precision))
    print('Micro recall:    {:0.2f}'.format(100 * micro_recall))
    print('Micro F1:        {:0.2f}'.format(100 * 2 * micro_precision * micro_recall / (micro_precision + micro_recall)))
    print()
