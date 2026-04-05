import json
import ast
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        self.invertedVocab = {}
        for tokenId, tokenBytes in self.vocab.items():
            self.invertedVocab[tokenBytes] = tokenId

        for specialToken in self.special_tokens:
            encodedSpecial = specialToken.encode("utf-8")
            if encodedSpecial not in self.invertedVocab:
                nextId = max(self.vocab.keys()) + 1
                self.vocab[nextId] = encodedSpecial
                self.invertedVocab[encodedSpecial] = nextId

        self.mergeRanks = {}
        for rank, (tokenA, tokenB) in enumerate(self.merges):
            self.mergeRanks[(tokenA, tokenB)] = rank

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            rawVocab = json.load(f)

        vocab = {}
        for tokenIdStr, byteValues in rawVocab.items():
            vocab[int(tokenIdStr)] = bytes(byteValues)

        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                splitIndex = line.index(" b'", 1)
                tokenA = ast.literal_eval(line[:splitIndex])
                tokenB = ast.literal_eval(line[splitIndex + 1:])
                merges.append((tokenA, tokenB))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        tokenIds = []

        if self.special_tokens:
            specialPattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            segments = re.split(f"({specialPattern})", text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue

            if segment in self.special_tokens:
                encodedSegment = segment.encode("utf-8")
                tokenIds.append(self.invertedVocab[encodedSegment])
                continue

            for match in re.finditer(PAT, segment):
                preToken = [bytes([b]) for b in match.group().encode("utf-8")]

                for mergeA, mergeB in self.merges:
                    newPreToken = []
                    i = 0
                    while i < len(preToken):
                        if i < len(preToken) - 1 and preToken[i] == mergeA and preToken[i + 1] == mergeB:
                            newPreToken.append(mergeA + mergeB)
                            i += 2
                        else:
                            newPreToken.append(preToken[i])
                            i += 1
                    preToken = newPreToken

                for token in preToken:
                    tokenIds.append(self.invertedVocab[token])

        return tokenIds

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for tokenId in self.encode(text):
                yield tokenId

    def decode(self, ids: list[int]) -> str:
        allBytes = b""
        for tokenId in ids:
            allBytes += self.vocab[tokenId]
        return allBytes.decode("utf-8", errors="replace")
    
