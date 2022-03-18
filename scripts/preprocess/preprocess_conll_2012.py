import argparse
import json
import logging
import os


def parse(path, keep_only_verb_predicates=False, keep_only_noun_predicates=False, keep_only_core_roles=False, add_predicate_pos=False):
    verb_pos_set = {'VBD', 'VBZ', 'VBG', 'VBP', 'VBN', 'MD', 'VB'}
    core_roles = {'A0', 'A1', 'A2', 'A3', 'A4', 'A5', '_'}
    data = {}

    with open(path) as f:
        sentence_index = 0
        sentence_words = []
        sentence_pos_tags = []
        sentence_predicates = []
        sentence_predicate_indices = []
        sentence_roles = []

        for line in f:
            line = line.strip()
            if line and line[0] == '#':
                continue

            if not line:
                sentence_roles = list(map(list, zip(*sentence_roles)))
                sentence_roles = sentence_roles[:-1]
                sentence_bio_roles = []
                for predicate_roles in sentence_roles:
                    bio_roles = []
                    current_role = '_'
                    inside_argument = False
                    for role in predicate_roles:
                        assert '*' in role, 'Error: found arg ({}) with no *.'.format(role)
                        assert role.count('(') < 2, '{}'.format(' '.join(sentence_words))
                        assert role.count(')') < 2, '{}'.format(' '.join(sentence_words))
                        if '(' in role and ')' in role:
                            current_role = role[1:-2]
                            inside_argument = False
                            bio_roles.append('B-{}'.format(current_role))
                            continue
                        if '(' in role:
                            current_role = role[1:-1]
                            inside_argument = True
                            bio_roles.append('B-{}'.format(current_role))
                            continue
                        if ')' in role:
                            inside_argument = False
                            bio_roles.append('I-{}'.format(current_role))
                            continue
                        if inside_argument:
                            bio_roles.append('I-{}'.format(current_role))
                        else:
                            bio_roles.append('_')
                    sentence_bio_roles.append(bio_roles)

                sentence_bio_roles = {idx: roles for idx, roles in zip(sentence_predicate_indices, sentence_bio_roles)}

                if keep_only_verb_predicates:
                    sentence_predicates = [p if sentence_pos_tags[p_i] in verb_pos_set else '_' for p_i, p in enumerate(sentence_predicates)]
                    sentence_bio_roles = {p_i: r for p_i, r in sentence_bio_roles.items() if sentence_pos_tags[p_i] in verb_pos_set}

                if keep_only_noun_predicates:
                    sentence_predicates = [p if sentence_pos_tags[p_i] not in verb_pos_set else '_' for p_i, p in enumerate(sentence_predicates)]
                    sentence_bio_roles = {p_i: r for p_i, r in sentence_bio_roles.items() if sentence_pos_tags[p_i] not in verb_pos_set}

                if keep_only_core_roles:
                    sentence_bio_roles = {p_i: [r if r == '_' or r[2:] in core_roles else '_' for r in roles] for p_i, roles in sentence_bio_roles.items()}

                sentence_data = {
                    'words': sentence_words,
                    'pos_tags': sentence_pos_tags,
                    'predicates': sentence_predicates,
                    'roles': sentence_bio_roles,
                }
                data[len(data)] = sentence_data

                sentence_index = 0
                sentence_words = []
                sentence_pos_tags = []
                sentence_predicates = []
                sentence_predicate_indices = []
                sentence_roles = []
                continue

            parts = line.split()

            word = parts[3].strip()
            sentence_words.append(word)

            pos_tag = parts[4].strip()
            sentence_pos_tags.append(pos_tag)

            predicate = parts[6].strip()
            predicate_sense = parts[7].strip()
            if predicate_sense != '-':
                sentence_predicate_indices.append(sentence_index)

                if add_predicate_pos and pos_tag.upper() in verb_pos_set:
                    predicate = '{}-v.{}'.format(predicate, predicate_sense)
                else:
                    predicate = '{}.{}'.format(predicate, predicate_sense)
                sentence_predicates.append(predicate)
            else:
                sentence_predicates.append('_')

            roles = parts[11:]
            sentence_roles.append(roles)
            sentence_index += 1

    return data


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
        '--keep_only_verb_predicates',
        action='store_true',
        help='Keep only the predicates tagged with a verbal POS tag. (English only)')
    parser.add_argument(
        '--keep_only_noun_predicates',
        action='store_true',
        help='Keep only the predicates tagged with a nominal POS tag. (English only)')
    parser.add_argument(
        '--keep_only_core_roles',
        action='store_true',
        help='Keep only those arguments tagged with core roles (e.g. A0, A1, A2, ...). (English only)')
    parser.add_argument(
        '--add_predicate_pos',
        action='store_true',
        help='Add a pos label to the predicate sense label (may be useful in English).')
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    logging.info('Parsing {}...'.format(args.input_path))

    parsed_data = parse(
        args.input_path,
        keep_only_verb_predicates=args.keep_only_verb_predicates,
        keep_only_noun_predicates=args.keep_only_noun_predicates,
        keep_only_core_roles=args.keep_only_core_roles,
        add_predicate_pos=args.add_predicate_pos)
    write_parsed_data(parsed_data, args.output_path)

    logging.info('Done!')
