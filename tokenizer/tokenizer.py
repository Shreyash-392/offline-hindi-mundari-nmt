import sentencepiece as spm


class HindiMundariTokenizer:
    def __init__(self, model_path="tokenizer/hindi_mundari.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = 3

    def encode(self, text, max_length=80):
        token_ids = self.sp.encode(text, out_type=int)

        # Add BOS and EOS
        token_ids = [self.bos_id] + token_ids + [self.eos_id]

        # Truncate
        token_ids = token_ids[:max_length]

        # Pad
        token_ids += [self.pad_id] * (max_length - len(token_ids))

        return token_ids

    def decode(self, token_ids):
        # Remove special tokens
        token_ids = [
            token_id
            for token_id in token_ids
            if token_id not in {
                self.pad_id,
                self.bos_id,
                self.eos_id,
            }
        ]

        return self.sp.decode(token_ids)