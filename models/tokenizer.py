import regex as re
from collections.abc import Iterable
import pickle


class Tokenizer():
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] = None,
                 pre_tokens_splitter: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):

        self.vocab = vocab
        self.merges = merges

        self.pre_tokens_splitter = pre_tokens_splitter

        # Build encoder dicts
        self.tokens_to_id = {self.vocab[key]: key for key in self.vocab.keys()}

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token in self.special_tokens:
                if self.tokens_to_id.get(bytes(special_token.encode('utf-8')), -1) == -1:
                    self.vocab[len(self.vocab)] = bytes(special_token.encode('utf-8'))
                    self.tokens_to_id[bytes(special_token.encode(
                        'utf-8'))] = len(self.tokens_to_id)
        else:
            self.special_tokens = special_tokens
        self.merges_to_order = {self.merges[i]: i for i in range(len(self.merges))}

    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] = None):
        """ Construct and return a Tokenizer from a serialized vocabulary and
            list of merges, and optionally a list of special tokens. """

        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """ Encode an input text into a sequence of token IDs. """

        if self.special_tokens:
            specials_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            # Ugly regex to prevent edge cases from test functions, I spent way too much time on this
            # If you find a better way to do this, please let me know
            pattern = rf"(?>{specials_pattern})|(?:.+?(?=(?:{specials_pattern})|$))|(?:{self.pre_tokens_splitter})"
            chunks = re.findall(pattern, text)
        else:
            chunks = [text]

        # Pre-tokenize the input text
        pre_tokens = []
        for chunk in chunks:
            if self.special_tokens and (chunk in self.special_tokens):
                pre_tokens.append([bytes(chunk.encode('utf-8'))])
            else:
                words = re.findall(self.pre_tokens_splitter, chunk)
                for word in words:
                    # Represent each pre-token as a tuple of individual bytes
                    pre_tokens.append([bytes([b]) for b in word.encode('utf-8')])

        # Encode the text
        encoded_text = []
        for pre_token in pre_tokens:

            # Merge pairs of tokens by order of creation during the training process
            while len(pre_token) > 1:
                best_merge = min([(self.merges_to_order.get(pair, float('inf')), i)
                                 for i, pair in enumerate(zip(pre_token[:-1], pre_token[1:]))])
                # The length of pre_token is more than 1 but can no longer be reduced
                if best_merge[0] == float('inf'):
                    break
                best_pair = self.merges[best_merge[0]]
                token_to_replace_id = best_merge[1]

                merge = best_pair[0] + best_pair[1]

                pre_token = (pre_token[:token_to_replace_id] +
                             [merge] +
                             pre_token[token_to_replace_id+2:])

            # Encode the merged pre-token into vocab IDs
            for token in pre_token:
                encoded_text.append(self.tokens_to_id[token])

        return encoded_text

    def encode_iterable(self, iterable: Iterable[str]):
        """ Given an iterable of strings, return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files that we cannot
        directly load into memory. """

        for chunks in iterable:
            for token_id in self.encode(chunks):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """ Decode a sequence of token IDs to the corresponding string """

        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors='replace')
