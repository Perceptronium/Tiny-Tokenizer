from collections import Counter, defaultdict
import regex as re
import os
from typing import BinaryIO
from multiprocessing import Pool


class BytePairEncoder():
    def __init__(self,
                 pre_tokens_splitter: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):

        self.pre_tokens_splitter = pre_tokens_splitter
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab: dict[int, bytes] = {}

    def initialize_vocab(self, special_tokens):
        """ Initialize the vocab with the 256 byte values and special tokens"""

        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, special_token in enumerate(special_tokens):
            self.vocab[256+i] = bytes(special_token.encode('utf-8'))
        return self.vocab

    def pre_tokenize_chunk(self, input_path, start, end, special_tokens):
        """ Pre-tokenize a chunk of text from the corpus. This function
            is sent to workers for parallel processing """

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        worker_pre_tokens_ctr = Counter()
        chunk = re.split("|".join(re.escape(tok)
                                  for tok in special_tokens), chunk)
        for sequence in chunk:
            words = re.findall(self.pre_tokens_splitter, sequence)
            worker_pre_tokens_ctr.update(words)
        return worker_pre_tokens_ctr

    def pre_tokenize_corpus(self, input_path, special_tokens):
        """ Pre-tokenize the whole corpus (parallelized)"""

        # Create as many processes as there are available cores in the CPU
        num_processes = os.cpu_count()
        with open(input_path, "rb") as f:
            # Find chunk boundaries (provided by CS-336)
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Parallelize the pre-tokenization step using DP-like parallelization: each process receives a different chunk of the corpus
        with Pool(num_processes) as workers:
            worker_args = [(input_path, start, end, special_tokens)
                           for (start, end) in zip(boundaries[:-1], boundaries[1:])]

            workers_pre_tokens_ctrs = workers.starmap(self.pre_tokenize_chunk,
                                                      worker_args)

        pre_tokens_ctr = Counter()
        for worker_pre_tokens_ctr in workers_pre_tokens_ctrs:
            pre_tokens_ctr.update(worker_pre_tokens_ctr)

        return pre_tokens_ctr

    def train_bpe(self,
                  input_path: str,
                  vocab_size: int,
                  special_tokens: list[str],
                  verbose: bool = False):

        # Initialize vocab with the 256 byte values and special tokens
        self.vocab = self.initialize_vocab(special_tokens)

        # Pre-tokenize the corpus (parallelized across CPU cores)
        pre_tokens_ctr = self.pre_tokenize_corpus(input_path, special_tokens)

        # Reformat the pre-tokens to store them as a dict[tuple[bytes], int]
        pre_tokens = {tuple(bytes([b]) for b in key.encode('utf-8')):
                      pre_tokens_ctr[key] for key in pre_tokens_ctr.keys()}

        # Cache the pair-counter: this is much faster than recounting the pairs after each merge
        pair_counter = Counter()
        # Use sets to avoid storing the same pre_token multiple times when building pair_contained_in
        pair_contained_in = defaultdict(set)
        for pre_token in pre_tokens:
            for pair in zip(pre_token[:-1], pre_token[1:]):
                pair_counter[pair] += pre_tokens[pre_token]
                # Cache the words where the pairs occur: this will be usefull during the merges
                pair_contained_in[pair].add(pre_token)

        # Main loop: building the vocab by merging the most frequent pairs
        while len(self.vocab) < vocab_size:

            if verbose:
                if len(self.vocab) % 100 == 0:
                    print(f"Vocab size {len(self.vocab)}")

            if len(pair_counter) == 0:
                print(
                    f"Encoded every pre-token into a token, returning vocab and merges.")
                return self.vocab, self.merges

            # Find the most frequent pair in the corpus
            most_frequent_pair = max(
                pair_counter, key=lambda p: (pair_counter[p], p))

            self.merges.append(most_frequent_pair)

            # Merge the most frequent pair
            new_token = most_frequent_pair[0] + most_frequent_pair[1]

            # Add it to the vocab
            self.vocab[len(self.vocab)] = new_token

            # Merge pre-tokens to take into account the new token
            # The naive way would be to scan every pre-token after each merge
            # What you really want is a direct pointer from most_frequent_pair to all pre_tokens where it appears
            pre_tokens_to_change = list(pair_contained_in[most_frequent_pair])
            for pre_token in pre_tokens_to_change:
                count = pre_tokens[pre_token]

                # Remove information from pair counter cache
                for pair in zip(pre_token[:-1], pre_token[1:]):
                    pair_counter[pair] -= count
                    pair_contained_in[pair].discard(pre_token)

                # Merge the pairs that match the most frequent one
                merged = []
                i = 0
                while i < len(pre_token):
                    if i + 1 < len(pre_token) and (pre_token[i], pre_token[i + 1]) == most_frequent_pair:
                        merged.append(new_token)
                        i += 2
                    else:
                        merged.append(pre_token[i])
                        i += 1
                merged = tuple(merged)

                # Add information to pair counter cache
                for pair in zip(merged[:-1], merged[1:]):
                    pair_counter[pair] += count
                    pair_contained_in[pair].add(merged)

                # Update the new merged pre-tokens dict
                pre_tokens[merged] = pre_tokens.get(merged, 0) + count
                del pre_tokens[pre_token]

        return self.vocab, self.merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    This is verbatim from CS 336.
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

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
