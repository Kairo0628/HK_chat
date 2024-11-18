from byte_pair_encoder import Tokenizer

import numpy as np

tokenizer = Tokenizer()

reserved_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '_']

sentences = ['안녕하세요 제 이름은 홍길동입니다.',
             '아 무한도전 진짜 ㅈㄴ 재밌네 ㅋㅋ',
             '현재 깃허브 리포지토리 주소는 https://github.com/Kairo0628/ml_dl_tutorials 입니다.',
             '전화번호는 010-1234-5678 입니다.',
             'kairo_o@naver.com 으로 문의 주세요.',
             '이 문장은 인코드 테스트용 문장입니다.']

tokenizer.fit(sentences = sentences,
              max_vocab_size = 2**10,
              reserved_tokens = reserved_tokens)

print(tokenizer._vocab)
print(tokenizer._vocab_index)
print(tokenizer.vocab_size)

print()

encoded = tokenizer.encode('현재 문장을 인코드 합니다. ㅋㅋ 번호는 010-1234-5678 입니다.')

print(encoded)

padded = tokenizer.vectorization(encoded, 30)
padded = np.array(padded)

print(padded)

print(tokenizer.decode(padded))
