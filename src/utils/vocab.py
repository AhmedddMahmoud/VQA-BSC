from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


_WORD_RE = re.compile(r"\w+|[^\w\s]")


class Vocab:
    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, token_to_idx: Dict[str, int]):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {index: token for token, index in token_to_idx.items()}
        self.pad_idx = token_to_idx[self.pad_token]
        self.unk_idx = token_to_idx[self.unk_token]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: Sequence[str], max_length: int) -> List[int]:
        ids = [self.token_to_idx.get(token, self.unk_idx) for token in tokens[:max_length]]
        if len(ids) < max_length:
            ids.extend([self.pad_idx] * (max_length - len(ids)))
        return ids

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.idx_to_token.get(index, self.unk_token) for index in ids]


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def tokenize_question(text: str) -> List[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _load_list_json(path: str | Path, key: str) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload[key]


def build_question_vocab(
    question_paths: Iterable[str | Path],
    max_vocab_size: int,
    min_freq: int = 1,
) -> Vocab:
    counter: Counter[str] = Counter()
    for path in question_paths:
        questions = _load_list_json(path, "questions")
        for item in questions:
            counter.update(tokenize_question(item["question"]))

    sorted_tokens = [token for token, count in counter.most_common() if count >= min_freq]
    final_tokens = [Vocab.pad_token, Vocab.unk_token] + sorted_tokens[: max_vocab_size - 2]
    token_to_idx = {token: index for index, token in enumerate(final_tokens)}
    return Vocab(token_to_idx)


def build_answer_vocab(
    annotation_path: str | Path,
    top_k: int,
) -> Tuple[Dict[str, int], List[str]]:
    annotations = _load_list_json(annotation_path, "annotations")
    counter: Counter[str] = Counter()

    for annotation in annotations:
        for answer_obj in annotation.get("answers", []):
            answer = normalize_answer(answer_obj["answer"])
            counter[answer] += 1

    top_answers = [answer for answer, _ in counter.most_common(top_k)]
    answer_to_idx = {answer: index for index, answer in enumerate(top_answers)}
    return answer_to_idx, top_answers


def extract_majority_answer(annotation: dict) -> str:
    answers = [normalize_answer(item["answer"]) for item in annotation.get("answers", [])]
    if not answers:
        return normalize_answer(annotation.get("multiple_choice_answer", ""))
    return Counter(answers).most_common(1)[0][0]
