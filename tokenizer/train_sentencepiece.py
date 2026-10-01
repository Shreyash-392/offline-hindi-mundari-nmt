import sentencepiece as spm
from pathlib import Path

INPUT_FILE = Path("tokenizer/train_text.txt")
OUTPUT_PREFIX = "tokenizer/hindi_mundari"

VOCAB_SIZE = 8000

spm.SentencePieceTrainer.train(
    input=str(INPUT_FILE),
    model_prefix=OUTPUT_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    input_sentence_size=0,
    shuffle_input_sentence=True,
)

print("SentencePiece tokenizer training complete")
print("-----------------------------------------")
print("Vocabulary size:", VOCAB_SIZE)
print("Model:", OUTPUT_PREFIX + ".model")
print("Vocabulary:", OUTPUT_PREFIX + ".vocab")