from tokenizers import Tokenizer
import os
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import json


def read_annotation(annotation_path):
    texts = []
    with open(annotation_path, encoding='utf-8') as f:
        for line in f:
            text = line.strip('\n').split('|')[1]
            texts.append(text)
    return texts


def combine_tokenizers(temp_folder):
    vocab = {}
    v = 0
    for i in ['old', 'new']:
        with open(temp_folder + i + '/vocab.json', encoding='utf-8') as f:
            data = json.load(f)
        for token in data.keys():
            if token in vocab:
                continue
            if i == 'old':
                assert data[token] == v
            vocab[token] = v
            v += 1
    with open(temp_folder + 'extended/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Keep duplicates in merges files
    os.system('cat {} > {}'.format(
        temp_folder + 'old/merges.txt',
        temp_folder + 'extended/merges.txt',
    ))
    os.system('tail -n +2 -q {} >> {}'.format(
        temp_folder + 'new/merges.txt',
        temp_folder + 'extended/merges.txt',
    ))


def extend_tokenizer(annotation_paths, special_tokens, old_vocab_path, new_vocab_path, extended_vocab_path, temp_folder, extended_vocab_size):
    for i in ['old', 'new', 'extended']:
        os.makedirs(temp_folder + i, exist_ok=True)

    # https://github.com/huggingface/tokenizers
    # https://huggingface.co/docs/tokenizers/index
    texts = [text for i in annotation_paths for text in read_annotation(i)]
    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=extended_vocab_size,
    )

    old_tokenizer = Tokenizer.from_file(old_vocab_path)
    old_tokenizer.model.save(temp_folder + 'old')

    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()
    new_tokenizer.train_from_iterator(texts, trainer=trainer)
    new_tokenizer.add_special_tokens(special_tokens)
    new_tokenizer.model.save(temp_folder + 'new')
    new_tokenizer.save(new_vocab_path)

    combine_tokenizers(temp_folder)

    old_tokenizer.model = old_tokenizer.model.from_file(
        temp_folder + 'extended/vocab.json',
        temp_folder + 'extended/merges.txt',
    )
    old_tokenizer.add_special_tokens(special_tokens)
    old_tokenizer.save(extended_vocab_path)


# 将 sample 替换为要训练的新语言的语言代码/名称
# models/metadata.txt 是数据集的转写文件
# original_model/vocab.json 是原始模型的词汇文件
# 后面 3 个 models 下的是新生成的文件
languages = ['sample']
extend_tokenizer(
    ['models/metadata.txt'],
    [f'[{language}]' for language in languages],
    'original_model/vocab.json',
    'models/vocab_new_only.json',
    'models/vocab.json',
    'models/temp_tokenizers/',
    2000,
)
