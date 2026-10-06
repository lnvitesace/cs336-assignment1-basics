from collections import Counter, defaultdict
import regex as re
import os
from typing import BinaryIO
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(chunk: str) -> Counter[tuple[bytes, ...]]:
    """Split text with the GPT-2 regex and count each pre-token as a tuple of single bytes."""
    counts = Counter()
    for m in re.finditer(PAT, chunk):
        tok = m.group().encode("utf8")
        counts[tuple(bytes([b]) for b in tok)] += 1
    return counts


def pre_tokenize_chunk(input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]) -> Counter:
    """
    Pre tokenize a file chunk and return the pre-token counts.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf8", errors="ignore")
    if special_tokens:
        split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(split_pattern, chunk)
    else:
        segments = [chunk]

    counts = Counter()
    for seg in segments:
        counts.update(pre_tokenize(seg))
    return counts


def pre_tokenize_file(
    input_path: str | os.PathLike, special_tokens: list[str], num_processes: int
) -> Counter[tuple[bytes, ...]]:
    """
    Pre-tokenize the whole file in parallel and return merged pre-token counts.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_processes) as pool:
        results = pool.starmap(pre_tokenize_chunk, args)

    counts = Counter()
    for c in results:
        counts.update(c)
    return counts


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pairs(
    counts: Counter[tuple[bytes, ...]],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    Count:
    1. pair_counts: Adjacent pairs within each pre-token, weighted by pre-token frequency.
    2. pair_to_words: A map from pair to words, which contain the pair
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for word, freq in counts.items():
        for pair in zip(word[:-1], word[1:]):
            pair_to_words[pair].add(word)
            pair_counts[pair] += freq

    return pair_counts, pair_to_words


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes], merged: bytes) -> tuple[bytes, ...]:
    """Replace every non-overlapping occurrence of `pair` in `word` with `merged`, left to right."""
    first, second = pair
    new_word, i, n = [], 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def apply_merge(
    counts: Counter[tuple[bytes, ...]],
    pair_counts: Counter[tuple[bytes, bytes]],
    merge: tuple[bytes, bytes],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> tuple[Counter[tuple[bytes, ...]], Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """Apply one merge to every pre-token, return the updated counts and pair_counts"""
    merged_pair = merge[0] + merge[1]
    for word in list(pair_to_words[merge]):
        if word not in counts:
            continue

        # Update counts
        freq = counts.pop(word)
        merged_word = merge_word(word, merge, merged_pair)
        counts[merged_word] += freq

        # Update pair_counts
        for pair in zip(word[:-1], word[1:]):
            pair_counts[pair] -= freq
            if pair_counts[pair] == 0:
                pair_counts.pop(pair)
        for pair in zip(merged_word[:-1], merged_word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(merged_word)

    return counts, pair_counts, pair_to_words


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train the BPE tokenizer
    """
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf8")
    merges: list[tuple[bytes, bytes]] = []

    counts = pre_tokenize_file(input_path, special_tokens, 4)
    pair_counts, pair_to_words = count_pairs(counts)

    # --- Merge until vocab size reached ---
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        merge = max(pair_counts, key=lambda k: (pair_counts[k], k))
        merges.append(merge)
        vocab[len(vocab)] = merge[0] + merge[1]
        counts, pair_counts, pair_to_words = apply_merge(counts, pair_counts, merge, pair_to_words)
    return vocab, merges


def train_bpe_tinystories():
    vocab, _ = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    print(vocab)


if __name__ == "__main__":
    import ast

    with open("vocab.txt") as f:
        vocab = ast.literal_eval(f.read())
    print(max(vocab.values(), key=lambda x: len(x)))
