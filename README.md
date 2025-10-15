# Tiny Tokenizer:
I wrote this self-contained implementation of a Byte-Pair Encoding (BPE) tokenizer covering the full pipeline from training to encoding / decoding.

My goal was essentially to understand how to properly build one so I tried to make the code as readable as possible for future reference, while keeping acceptable performance. It's decent but can still be improved.

**NB:** This is essentially a dressed-up version of my resolution of the tokenizer-from-scratch task of Stanford's CS-336.

<p align="center">
  <table>
    <tr>
      <td align="center"><h2><b>Tiktoken</b> <br/>(OpenAI)</h2></td>
      <td align="center"><h2><b>Tiny Tokenizer</b> <br/>(this repo)</h2></td>
    </tr>
    <tr>
      <td align="center"><img src="./illus/repo_illu.png" width="220"/></td>
      <td align="center"><img src="./illus/repo_illu_2.png" width="220"/></td>
    </tr>
  </table>
</p>

---

#### The (naive) BPE algorithm in a nutshell:
1. On the corpus to train on, count adjacent pairs of bytes and find the most frequent one.
2. Merge that pair into a new vocab entry.
3. Update the corpus to reflect this change.
4. Repeat steps 1–3 until a target vocabulary size is reached or no more merge improves compression.


#### Additional features included in this implementation:
- Parallelized pre-tokenization (uses maximum amount of available CPU cores).
- In-place update of pre-tokens during merges.
- Handle user-specified special tokens.

---
#### Repo structure:

- `models/byte_pair_encoder.py` — defines the `BytePairEncoder` class with a `train_bpe` method that builds a vocab to a desired target size.
- `models/tokenizer.py` — defines the `Tokenizer` class that loads the trained vocab and associated merges, and provides `encode` / `decode` methods.
- `train_bpe.py` — CLI to train and save `vocab.pkl` and `merges.pkl`.
- `test_tokenizer.py` — CLI to encode/decode sample inputs using saved vocab and merges.

---

## Example usage:

Download data:
```
mkdir -p data
cd data
```
TinyStories dataset (smaller one):

```
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```
OpenWebText dataset (bigger one):
```
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

Train an encoder, eg:
```
python train_bpe.py -d ./data/TinyStoriesV2-GPT4-train.txt -v 10000 --special-token '<|endoftext|>' --verbose
```

Encode or decode user specified input through the CLI, eg:
```
python test_tokenizer.py --vocab ./results/TinyStoriesV2-GPT4-train_bpe_vocab.pkl --merges ./results/TinyStoriesV2-GPT4-train_bpe_merges.pkl --encode "Hello, how are you?"

python test_tokenizer.py --vocab ./results/TinyStoriesV2-GPT4-train_bpe_vocab.pkl --merges ./results/TinyStoriesV2-GPT4-train_bpe_merges.pkl --decode "1183, 44, 763, 483, 349, 63"
```
