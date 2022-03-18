import json

from torch.utils.data import Dataset


class ConllDataset(Dataset):

    def __init__(self, path_to_data, span_based=False):
        super().__init__()
        self.sentences = ConllDataset.load_sentences(path_to_data, span_based=span_based)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    @staticmethod
    def load_sentences(path, span_based=False):
        sentences = []

        with open(path) as json_file:

            for i, sentence in json.load(json_file).items():
                if len(sentence['words']) > 128:
                    continue
                if not [p for p in sentence['predicates'] if p != '_']:
                    continue
                
                if span_based:
                    rolesets = {
                        int(predicate_index): set([r for r in roles if ConllDataset.is_valid_role(r[2:]) and r != '_' and r[:2] != 'I-'])
                        for predicate_index, roles in sentence['roles'].items()
                    }
                else:
                    rolesets = {
                        int(predicate_index): set([r for r in roles if ConllDataset.is_valid_role(r)])
                        for predicate_index, roles in sentence['roles'].items()
                    }
                
                sentences.append({
                    'sentence_id': int(i),
                    'words': ConllDataset.process_words(sentence['words']),
                    'predicates': sentence['predicates'],
                    'rolesets': rolesets,
                    'roles': {
                        int(predicate_index): roles
                        for predicate_index, roles in sentence['roles'].items()
                    },
                })

        return sentences

    @staticmethod
    def process_words(words):
        processed_words = []
        for word in words:
            if word == '-LRB-' or word == '-LSB-':
                processed_word = '('
            elif word == '-RRB-' or word == '-RSB-':
                processed_word = ')'
            elif word == '-LCB-' or word == '-RCB-':
                processed_word = ''
            elif word == '``' or word == "''":
                processed_word = '"'
            else:
                processed_word = word
            processed_words.append(processed_word)
        return processed_words

    @staticmethod
    def is_valid_role(role):
        if role == '_':
            return False
        if role == 'V':
            return False
        if role[:2] == 'C-' or role[:2] == 'R-':
            return False
        return True