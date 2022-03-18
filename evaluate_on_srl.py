import subprocess
from argparse import ArgumentParser

import torch

from multilingual_srl.data.conll_data_module import ConllDataModule
from multilingual_srl.models.srl_model import SrlModel

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--scorer', type=str, default='scripts/evaluation/scorer_conll2009.pl')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--datamodule', type=str, required=True)
    parser.add_argument('--conll_test', type=str, required=True)
    parser.add_argument('--gold_output', type=str, required=True)
    parser.add_argument('--pred_output', type=str, required=True)

    # Add data-specific args.
    parser = ConllDataModule.add_data_specific_args(parser)

    # Store the arguments in hparams.
    args = parser.parse_args()

    data_module = ConllDataModule.load_from_config(args.datamodule)
    data_module.test_path = args.test_path
    data_module.batch_size = args.batch_size
    data_module.num_workers = args.num_workers
    data_module.prepare_data()
    data_module.setup('test')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SrlModel.load_from_checkpoint(args.checkpoint)
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
                    'roles': batch_predictions['roles'][i],
                }

    sentence_id = 0
    sentence_output = []
    sentence_senses = []
    sentence_roles = []
    with open(args.conll_test) as f_in, open(args.gold_output, 'w') as f_gold_out, open(args.pred_output, 'w') as f_pred_out:
        for line in f_in:
            line = line.strip()
            if not line:
                if sentence_id not in predictions:
                    for i in range(len(sentence_output)):
                        sentence_output[i][12] = '_'
                        output_line = '\t'.join(sentence_output[i])
                        f_pred_out.write('{}\t_\n'.format(output_line))
                        f_gold_out.write('{}\t_\n'.format(output_line))
                    f_pred_out.write('\n')
                    f_gold_out.write('\n')
                    sentence_id += 1
                    sentence_output = []
                    sentence_senses = []
                    sentence_roles = []
                    continue

                sentence_roles = list(map(list, zip(*sentence_roles)))
                predicted_senses = predictions[sentence_id]['senses']
                output_senses = []
                num_predicates = 0
                for predicate_index in range(len(sentence_senses)):
                    gold = sentence_senses[predicate_index]
                    predicted = predicted_senses[predicate_index] if predicate_index in predicted_senses else '_'

                    if predicted == '_' and gold != '_':
                        sentence_output[predicate_index][12] = '_'
                        sentence_senses[predicate_index] = '_'
                        sentence_roles[num_predicates] = []
                    
                    if predicted != '_':
                        output_senses.append(predicted)
                    else:
                        output_senses.append('_')
                    
                    if gold != '_':
                        num_predicates += 1

                predicted_roles = predictions[sentence_id]['roles']
                output_roles = []
                for i in range(len(sentence_senses)):
                    if output_senses[i] != '_':
                        output_roles.append(predicted_roles[i])

                output_roles = list(map(list, zip(*output_roles)))
                sentence_roles = [r for r in sentence_roles if r]
                sentence_roles = list(map(list, zip(*sentence_roles)))
                for i in range(len(sentence_output)):
                    if output_roles:
                        pred_line_parts = sentence_output[i] + [output_senses[i]] + output_roles[i]
                        gold_line_parts = sentence_output[i] + [sentence_senses[i]] + sentence_roles[i]
                    else:
                        pred_line_parts = sentence_output[i] + [output_senses[i]]
                        gold_line_parts = sentence_output[i] + sentence_senses[i]
                    pred_output_line = '\t'.join(pred_line_parts)
                    gold_output_line = '\t'.join(gold_line_parts)
                    f_pred_out.write('{}\n'.format(pred_output_line))
                    f_gold_out.write('{}\n'.format(gold_output_line))
                f_pred_out.write('\n')
                f_gold_out.write('\n')

                sentence_id += 1
                sentence_output = []
                sentence_senses = []
                sentence_roles = []
                continue

            parts = line.split('\t')
            sentence_output.append(parts[:13])
            sentence_senses.append(parts[13])
            sentence_roles.append(parts[14:])

    subprocess.run(['perl', args.scorer, '-g', args.gold_output, '-s', args.pred_output, '-q'])
