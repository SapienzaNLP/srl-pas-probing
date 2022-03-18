import argparse
import json
import logging
import os


def parse(path):
    with open(path) as f:
        data = json.load(f)
    
    predicate_pos = {}
    for sentence_id, sentence in data.items():
        for predicate, pos in zip(sentence['predicates'], sentence['pos_tags']):
            if pos not in predicate_pos:
                predicate_pos[pos] = 0
            if predicate != '_':
                predicate_pos[pos] += 1
    
    return predicate_pos



def write_parsed_data(data, path):
    output = json.dumps(data, indent=4, sort_keys=True)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        dest='input_path',
        help='Path to the data to preprocess.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path to the output file.')
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    logging.info('Parsing {}...'.format(args.input_path))

    parsed_data = parse(args.input_path)
    write_parsed_data(parsed_data, args.output_path)

    logging.info('Done!')