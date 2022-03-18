import argparse
import json
import logging
import os


def parse(path, czech=False, keep_only_verb_predicates=False, keep_only_noun_predicates=False, keep_only_core_roles=False, keep_pos_tags=False, keep_lemmas=False, add_predicate_pos=False):
    verb_pos_set = {'VBD', 'VBZ', 'VBG', 'VBP', 'VBN', 'MD', 'VB'}
    core_roles = {'A0', 'A1', 'A2', 'A3', 'A4', 'A5', '_'}
    data = {}

    assert not(keep_only_verb_predicates and keep_only_noun_predicates)

    with open(path) as f:
        word_index = 0
        sentence_words = []
        sentence_lemmas = []
        sentence_pos_tags = []
        sentence_predicates = []
        sentence_predicate_indices = []
        sentence_roles = []

        for line in f:
            line = line.strip()
            if not line:
                sentence_roles = list(zip(*sentence_roles))
                sentence_roles = {idx: roles for idx, roles in zip(sentence_predicate_indices, sentence_roles)}

                if keep_only_verb_predicates:
                    sentence_predicates = [p if sentence_pos_tags[p_i] in verb_pos_set else '_' for p_i, p in enumerate(sentence_predicates)]
                    sentence_roles = {p_i: r for p_i, r in sentence_roles.items() if sentence_pos_tags[p_i] in verb_pos_set}

                if keep_only_noun_predicates:
                    sentence_predicates = [p if sentence_pos_tags[p_i] not in verb_pos_set else '_' for p_i, p in enumerate(sentence_predicates)]
                    sentence_roles = {p_i: r for p_i, r in sentence_roles.items() if sentence_pos_tags[p_i] not in verb_pos_set}

                if keep_only_core_roles:
                    sentence_roles = {p_i: [r if r in core_roles else '_' for r in roles] for p_i, roles in sentence_roles.items()}

                sentence_data = {
                    'words': sentence_words,
                    'predicates': sentence_predicates,
                    'roles': sentence_roles,
                }

                if keep_pos_tags:
                    sentence_data['pos_tags'] = sentence_pos_tags
                if keep_lemmas:
                    sentence_data['lemmas'] = sentence_lemmas

                data[len(data)] = sentence_data

                word_index = 0
                sentence_words = []
                sentence_lemmas = []
                sentence_pos_tags = []
                sentence_predicates = []
                sentence_predicate_indices = []
                sentence_roles = []
                continue

            parts = line.split('\t')

            word = parts[1].strip()
            sentence_words.append(word)

            lemma = parts[3].strip()
            sentence_lemmas.append(lemma)

            pos_tag = parts[4].strip()
            sentence_pos_tags.append(pos_tag)

            predicate = parts[13].strip()
            if predicate != '_':
                sentence_predicate_indices.append(word_index)

                if add_predicate_pos and pos_tag.upper() in verb_pos_set:
                    predicate_lemma, predicate_number = predicate.split('.')
                    predicate = '{}-v.{}'.format(predicate_lemma, predicate_number)

                if czech and predicate[:3] != 'v-w':
                    sentence_predicates.append('lemma')
                else:
                    sentence_predicates.append(predicate)

            else:
                sentence_predicates.append(predicate)

            roles = parts[14:]
            sentence_roles.append(roles)
            word_index += 1

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
        '--keep_pos_tags',
        action='store_true',
        help='Keep the POS tags when parsing the dataset.')
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
        '--keep_lemmas',
        action='store_true',
        help='Keep the lemmas when parsing the dataset.')
    parser.add_argument(
        '--add_predicate_pos',
        action='store_true',
        help='Add a pos label to the predicate sense label (may be useful in English).')
    parser.add_argument(
        '--czech',
        action='store_true',
        help='For the Czech dataset.')
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
        czech=args.czech,
        keep_only_verb_predicates=args.keep_only_verb_predicates,
        keep_only_noun_predicates=args.keep_only_noun_predicates,
        keep_only_core_roles=args.keep_only_core_roles,
        keep_pos_tags=args.keep_pos_tags,
        keep_lemmas=args.keep_lemmas,
        add_predicate_pos=args.add_predicate_pos)

    write_parsed_data(parsed_data, args.output_path)

    logging.info('Done!')