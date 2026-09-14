"""Split by canonical puzzle so duplicates/permutations never cross the holdout."""
import hashlib
import json
import random


def puzzle_key(row):
    return (int(row["target"]), tuple(sorted(int(n) for n in row["nums"])))


def held_out(key, seed, fraction):
    digest = hashlib.sha256(json.dumps([seed, key]).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def split_rows(rows, config):
    train, evaluation, seen = [], [], set()
    for row in rows:
        key = puzzle_key(row)
        if key in seen:
            continue
        seen.add(key)
        normalized = {"target": key[0], "nums": [int(n) for n in row["nums"]]}
        (evaluation if held_out(key, config.seed, config.eval_fraction) else train).append(normalized)
    # Sort before shuffling to make membership independent of upstream row order.
    train.sort(key=puzzle_key)
    evaluation.sort(key=puzzle_key)
    random.Random(config.seed).shuffle(train)
    random.Random(config.seed).shuffle(evaluation)
    if not train or not evaluation:
        raise ValueError("Dataset must yield nonempty training and evaluation splits")
    return train[:config.train_samples], evaluation[:config.eval_samples]


def prompt(row):
    return [{"role": "user", "content": (
        f"Use the numbers {row['nums']} exactly once each to make {row['target']}. "
        "Use only +, -, *, / and parentheses. Fractions and negative intermediate "
        "results are allowed. You may work through the problem before answering. "
        "End with exactly one <answer>expression</answer> block. "
        "Put only the arithmetic expression inside the block, without an equals sign."
    )}]


def load_rows(config):
    from datasets import load_dataset
    dataset = load_dataset(config.dataset, revision=config.dataset_revision, split="train")
    return split_rows(dataset, config)


def render_prompt(tokenizer, row):
    # Render once as plain text so TRL and evaluation use identical templates.
    return tokenizer.apply_chat_template(prompt(row), tokenize=False, add_generation_prompt=True)
