import re
from collections import defaultdict
from itertools import combinations
import json

_phone_pattern = re.compile(r'\d{2,3}-\d{3,4}-\d{4}')
_email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_url_pattern = re.compile(r'https?://[^\s]+')
_consonant_pattern = re.compile(r'[ㄱ-ㅎ]')
_character_pattern = re.compile(r'[^a-zA-Z0-9가-힣ㄱ-ㅎ]')

class Tokenizer:
  def __init__(self):
    pass

  def fit(self, sentences, max_vocab_size, reserved_tokens):
    token_counts = defaultdict(int)
    phones = set()
    emails = set()
    urls = set()
    consonants = set()
    characters = set()

    for i in sentences:
      phone, email, url, consonant, character = _find_patterns(i)

      phones.update(phone)
      emails.update(email)
      urls.update(url)
      consonants.update(consonant)
      characters.update(character)

      subwords = _remove_patterns(i)

      for subword in subwords:
        for start in range(len(subword)):
          for end in range(len(subword), start, -1):
            if len(subword[start:end]) > 3:
              continue
            token_counts[subword[start:end]] += 1

    phones = _prepare_before_merge(phones)
    emails = _prepare_before_merge(emails)
    urls = _prepare_before_merge(urls)
    consonants = _prepare_before_merge(consonants)
    characters = _prepare_before_merge(characters)
    reserved_tokens = _prepare_before_merge(reserved_tokens)
    specials = reserved_tokens | phones | emails | urls | consonants | characters

    self._vocab = sorted(token_counts.items(), key=lambda item: item[1])
    if max_vocab_size < len(specials) + len(self._vocab):
      self._vocab = self._vocab[len(specials) + len(self._vocab) - max_vocab_size:]

    self._vocab = dict(self._vocab)
    self._vocab = specials | self._vocab
    self._vocab_index = {j: i for i, j in enumerate(self._vocab)}
    self._reverse_index = {j: i for i, j in self._vocab_index.items()}

  @property
  def vocab_size(self):
    return len(self._vocab)

  def encode(self, sentence):
    tokens = sentence.replace(' ', ' _ ').split()
    pass_tokens = set()

    for pattern in [_phone_pattern, _email_pattern, _url_pattern, _consonant_pattern, _character_pattern]:
      for match in pattern.finditer(sentence):
        if match.group() in pass_tokens:
          continue
        pass_tokens.add(match.group())
        for idx, token in enumerate(tokens):
          if match.group() in token:
            if token in pass_tokens:
              continue
            start = token.index(match.group())
            end = token.index(match.group()) + len(match.group())
            new_tokens = token[:start] + ' ' + token[start:end] + ' ' + token[end:]
            tokens[idx:idx+1] = new_tokens.split()

    def _get_splits(token):
      if token in self._vocab:
        return [token]

      n = len(token)
      if n == 1:
        return ['<unk>']

      splits = []

      for i in range(1, n):
          for comb in combinations(range(1, n), i):
              parts = []
              prev = 0
              for j in comb:
                  parts.append(token[prev:j])
                  prev = j
              parts.append(token[prev:])
              splits.append(tuple(parts))

      return splits

    def _find_min_score(splits):
      n = len(splits)
      if n == 1:
        return self._vocab_index[splits[0]] if splits[0] in self._vocab else self._vocab_index['<unk>']

      scores = [0] * len(splits)

      for idx, tup in enumerate(splits):
        for i in tup:
          if i in self._vocab:
            scores[idx] += self._vocab[i]
          else:
            scores[idx] = -1
            break

      temp = [i for i in scores if i != -1]
      min_score = min(temp) if temp else -1

      if min_score == -1:
        return self._vocab_index['<unk>']
      else:
        best = splits[scores.index(min_score)]
        best_idx = []
        for i in best:
          best_idx.append(self._vocab_index[i])
        return best_idx

    index = []
    for i in tokens:
      splits = _get_splits(i)
      best_idx = _find_min_score(splits)
      if type(best_idx) == list:
        index += best_idx
      else:
        index.append(best_idx)

    return index

  def decode(self, sentence):
    text = ''

    for i in sentence:
      if self._reverse_index[i] == '<bos>':
        continue

      if self._reverse_index[i] == '<eos>':
        break

      text += self._reverse_index[i]

    text = text.replace('_', ' ')

    return text

  def vectorization(self, sentence, max_sentence_length):
    prepared_sentence = sentence
    prepared_sentence.insert(0, self._vocab_index['<bos>'])
    prepared_sentence.append(self._vocab_index['<eos>'])
    if len(prepared_sentence) > max_sentence_length:
      raise ValueError('max_sentence_length 가 문장의 길이보다 적어도 2 더 커야 합니다.')
    else:
      prepared_sentence += [self._vocab_index['<pad>']] * (max_sentence_length - len(prepared_sentence))

    return prepared_sentence

  def save_vocab(self, path = ''):
    with open(path + 'vocab.json', 'w', encoding = 'utf-8') as f:
      json.dump(self._vocab, f, ensure_ascii = False)

    print('Save Complete')

  def load_vocab(self, path):
    with open(path, 'r', encoding = 'utf-8') as f:
        vocab = json.load(f)

    self._vocab = {}
    for i, j in vocab.items():
      self._vocab[i] = j
    self._vocab_index = {j: i for i, j in enumerate(self._vocab)}
    self._reverse_index = {j: i for i, j in self._vocab_index.items()}

    print('Load Complete')

def _find_patterns(sentence):
  phone = _phone_pattern.findall(sentence)
  email = _email_pattern.findall(sentence)
  url = _url_pattern.findall(sentence)
  consonant = _consonant_pattern.findall(sentence)

  sentence = _phone_pattern.sub('', sentence)
  sentence = _email_pattern.sub('', sentence)
  sentence = _url_pattern.sub('', sentence)
  character = _character_pattern.findall(sentence)
  character = [i for i in character if i != ' ']

  return phone, email, url, consonant, character

def _remove_patterns(sentence):
  sentence = _phone_pattern.sub('', sentence)
  sentence = _email_pattern.sub('', sentence)
  sentence = _url_pattern.sub('', sentence)
  sentence = _consonant_pattern.sub('', sentence)
  subwords = _character_pattern.split(sentence)

  return [i for i in subwords if i]

def _prepare_before_merge(special_tokens):
  d = {}
  for i in special_tokens:
    d[i] = 1

  return d