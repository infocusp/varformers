from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer

def get_all_sentences(ds, col):
    for item in ds[col]:
        yield item

def get_build_tokenizer(filepath, ds, col):
    tokenizer_path = Path(filepath)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = BertPreTokenizer()
        trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, col), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer