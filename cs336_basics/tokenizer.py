from collections.abc import Iterable, Iterator
import regex as re
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """Construct a tokenizer from a givenvocabulary, list of merges, and (optionally) a list of special tokens."""
        self.vocab = dict(vocab)
        self.vocab_to_ids = {b: i for i, b in self.vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.vocab_to_ids:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = b
                self.vocab_to_ids[b] = new_id

        self.special_sets = set(self.special_tokens)

        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
        else:
            self.special_pattern = None

        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method
        that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _split_special(self, text):
        """Split text based on special tokens"""
        if self.special_pattern is None:
            return [text]
        return [p for p in self.special_pattern.split(text) if p]

    def _merge_tokens(self, tokens: list[bytes]):
        """Merge a word's bytes based on vocab and merges"""
        while len(tokens) > 1:
            best_pair, best_rank = None, None
            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            a, b = best_pair
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        ids = []

        for part in self._split_special(text):
            if part in self.special_sets:
                ids.append(self.vocab_to_ids[part.encode("utf8")])
                continue
            for m in re.finditer(PAT, part):
                word = m.group()
                cached = self.cache.get(word)
                if cached is None:
                    tokens = [bytes([b]) for b in word.encode("utf8")]
                    cached = [self.vocab_to_ids[t] for t in self._merge_tokens(tokens)]
                    self.cache[word] = cached
                ids.extend(cached)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """A generator that lazily yields encoded token IDs"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf8", errors="replace")
