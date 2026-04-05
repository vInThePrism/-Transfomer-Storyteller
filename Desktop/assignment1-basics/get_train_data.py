from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from typing import List
from tokenizers import Tokenizer


def load_tokenizer(tokenizer_json_path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(tokenizer_json_path)
    return tokenizer


def text_to_token_ids(
    text: str,
    tokenizer: Tokenizer,
    add_eos: bool = True
    ) -> List[int]:
    encoding = tokenizer.encode(text)
    ids = encoding.ids

    if add_eos:
        eos_id = tokenizer.token_to_id("<|eos|>")
        if eos_id is not None:
            ids.append(eos_id)

    return ids


def build_random_data_bin(
        input_txt: str,
        tokenizer_json: str,
        output_bin: str,
        target_samples: int = 10000,
        dtype=np.int32
):
    tokenizer = load_tokenizer(tokenizer_json)

    print(f"Scanning total line count...")
    with open(input_txt, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"Total {total_lines} lines, sampling {target_samples}...")

    # Sort sampled indices to allow streaming read
    target_indices = set(random.sample(range(total_lines), min(target_samples, total_lines)))

    all_ids = []

    with open(input_txt, "r", encoding="utf-8") as f:
        sampled_count = 0
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
            if i in target_indices:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                ids = text_to_token_ids(line_stripped, tokenizer)
                all_ids.extend(ids)

                sampled_count += 1
                if sampled_count >= target_samples:
                    break

    arr = np.array(all_ids, dtype=dtype)
    arr.tofile(output_bin)
    print(f"Saved {len(arr):,} tokens to {output_bin}")


if __name__ == "__main__":

    tokenizer_json_path = "bpe_tokenizer/tokenizer.json"

    build_random_data_bin(
        input_txt="TinyStoriesV2-GPT4-train.txt",
        tokenizer_json=tokenizer_json_path,
        output_bin="data.bin",
        target_samples=800000
    )
