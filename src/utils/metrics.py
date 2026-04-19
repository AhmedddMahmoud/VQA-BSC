from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from src.utils.vocab import normalize_answer


def normalize_answer_for_eval(answer: str) -> str:
    if not isinstance(answer, str):
        raise TypeError(f"answer must be str, got {type(answer)}")
    return normalize_answer(answer)


def exact_match_after_normalization(pred_answer: str, gold_answer: str) -> bool:
    return normalize_answer_for_eval(pred_answer) == normalize_answer_for_eval(gold_answer)


def indices_to_answers(indices: Sequence[int], answer_vocab: Sequence[str]) -> List[str]:
    answers: List[str] = []
    vocab_size = len(answer_vocab)
    for index in indices:
        if not (0 <= int(index) < vocab_size):
            raise ValueError(f"Index {index} out of range for answer_vocab size {vocab_size}")
        answers.append(answer_vocab[int(index)])
    return answers


def _to_python_list(values: Sequence[Any] | torch.Tensor) -> List[Any]:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def compute_top1_accuracy(pred_indices: Sequence[int] | torch.Tensor, gold_indices: Sequence[int] | torch.Tensor) -> float:
    pred_list = _to_python_list(pred_indices)
    gold_list = _to_python_list(gold_indices)

    if len(pred_list) != len(gold_list):
        raise ValueError(
            f"pred_indices and gold_indices length mismatch: {len(pred_list)} vs {len(gold_list)}"
        )
    if len(pred_list) == 0:
        return 0.0

    correct = 0
    for pred, gold in zip(pred_list, gold_list):
        if int(pred) == int(gold):
            correct += 1
    return correct / len(pred_list)


def compute_vqa_soft_accuracy(pred_answers: Sequence[str], all_gold_answers: Sequence[Sequence[str]]) -> float:
    if len(pred_answers) != len(all_gold_answers):
        raise ValueError(
            f"pred_answers and all_gold_answers length mismatch: {len(pred_answers)} vs {len(all_gold_answers)}"
        )
    if len(pred_answers) == 0:
        return 0.0

    total_score = 0.0
    for pred, gold_answers in zip(pred_answers, all_gold_answers):
        if not isinstance(gold_answers, Sequence) or isinstance(gold_answers, str):
            raise TypeError("Each item in all_gold_answers must be a sequence of strings")

        norm_pred = normalize_answer_for_eval(pred)
        norm_gold = [normalize_answer_for_eval(answer) for answer in gold_answers]
        match_count = sum(1 for answer in norm_gold if answer == norm_pred)
        sample_score = min(match_count / 3.0, 1.0)
        total_score += sample_score

    return total_score / len(pred_answers)


def _safe_tokenize_for_overlap(text: str) -> List[str]:
    normalized = normalize_answer_for_eval(text)
    if normalized == "":
        return []
    return normalized.split()


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    if len(tokens) < n:
        return Counter()
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return Counter(grams)


def _closest_reference_length(candidate_len: int, references: Sequence[Sequence[str]]) -> int:
    ref_lengths = [len(reference) for reference in references]
    if len(ref_lengths) == 0:
        return 0
    return min(ref_lengths, key=lambda length: (abs(length - candidate_len), length))


def compute_corpus_bleu(
    pred_answers: Sequence[str],
    all_gold_answers: Sequence[Sequence[str]],
    max_n: int,
    epsilon: float = 1e-9,
) -> float:
    if max_n <= 0:
        raise ValueError(f"max_n must be >= 1, got {max_n}")
    if len(pred_answers) != len(all_gold_answers):
        raise ValueError(
            f"pred_answers and all_gold_answers length mismatch: {len(pred_answers)} vs {len(all_gold_answers)}"
        )
    if len(pred_answers) == 0:
        return 0.0

    clipped_matches = [0.0 for _ in range(max_n)]
    total_candidates = [0.0 for _ in range(max_n)]

    total_candidate_length = 0
    total_reference_length = 0

    for pred_answer, references in zip(pred_answers, all_gold_answers):
        if not isinstance(references, Sequence) or isinstance(references, str):
            raise TypeError("Each item in all_gold_answers must be a sequence of strings")

        candidate_tokens = _safe_tokenize_for_overlap(pred_answer)
        reference_tokens_list = [_safe_tokenize_for_overlap(reference) for reference in references if isinstance(reference, str)]
        if len(reference_tokens_list) == 0:
            reference_tokens_list = [[]]

        total_candidate_length += len(candidate_tokens)
        total_reference_length += _closest_reference_length(len(candidate_tokens), reference_tokens_list)

        for n in range(1, max_n + 1):
            candidate_counts = _ngram_counts(candidate_tokens, n)
            candidate_total = sum(candidate_counts.values())
            total_candidates[n - 1] += candidate_total

            if candidate_total == 0:
                continue

            max_ref_counts: Counter = Counter()
            for reference_tokens in reference_tokens_list:
                ref_counts = _ngram_counts(reference_tokens, n)
                for gram, count in ref_counts.items():
                    if count > max_ref_counts[gram]:
                        max_ref_counts[gram] = count

            clipped = 0
            for gram, count in candidate_counts.items():
                clipped += min(count, max_ref_counts.get(gram, 0))
            clipped_matches[n - 1] += clipped

    if total_candidate_length == 0:
        return 0.0

    if total_candidate_length > total_reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (total_reference_length / total_candidate_length))

    precisions: List[float] = []
    for index in range(max_n):
        if total_candidates[index] == 0:
            precisions.append(epsilon)
        else:
            precision = clipped_matches[index] / total_candidates[index]
            precisions.append(max(precision, epsilon))

    log_precision_sum = sum(math.log(value) for value in precisions) / max_n
    bleu = brevity_penalty * math.exp(log_precision_sum)
    return float(bleu)


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    if len(first) == 0 or len(second) == 0:
        return 0

    previous = [0] * (len(second) + 1)
    for token in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def compute_rouge_l(
    pred_answers: Sequence[str],
    all_gold_answers: Sequence[Sequence[str]],
) -> float:
    if len(pred_answers) != len(all_gold_answers):
        raise ValueError(
            f"pred_answers and all_gold_answers length mismatch: {len(pred_answers)} vs {len(all_gold_answers)}"
        )
    if len(pred_answers) == 0:
        return 0.0

    sample_scores: List[float] = []
    for pred_answer, references in zip(pred_answers, all_gold_answers):
        candidate_tokens = _safe_tokenize_for_overlap(pred_answer)
        reference_tokens_list = [_safe_tokenize_for_overlap(reference) for reference in references if isinstance(reference, str)]
        if len(reference_tokens_list) == 0:
            reference_tokens_list = [[]]

        best_f1 = 0.0
        for reference_tokens in reference_tokens_list:
            lcs = _lcs_length(candidate_tokens, reference_tokens)
            if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
                f1 = 0.0
            else:
                precision = lcs / len(candidate_tokens)
                recall = lcs / len(reference_tokens)
                f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1

        sample_scores.append(best_f1)

    return float(sum(sample_scores) / len(sample_scores))


def _meteor_chunk_count(candidate_tokens: Sequence[str], reference_tokens: Sequence[str]) -> int:
    reference_positions: Dict[str, List[int]] = {}
    for index, token in enumerate(reference_tokens):
        reference_positions.setdefault(token, []).append(index)

    used_positions = set()
    aligned_positions: List[int] = []
    for token in candidate_tokens:
        positions = reference_positions.get(token, [])
        chosen = None
        for position in positions:
            if position not in used_positions:
                chosen = position
                break
        if chosen is not None:
            used_positions.add(chosen)
            aligned_positions.append(chosen)

    if len(aligned_positions) == 0:
        return 0

    chunks = 1
    for previous, current in zip(aligned_positions, aligned_positions[1:]):
        if current != previous + 1:
            chunks += 1
    return chunks


def compute_meteor(
    pred_answers: Sequence[str],
    all_gold_answers: Sequence[Sequence[str]],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    if len(pred_answers) != len(all_gold_answers):
        raise ValueError(
            f"pred_answers and all_gold_answers length mismatch: {len(pred_answers)} vs {len(all_gold_answers)}"
        )
    if len(pred_answers) == 0:
        return 0.0

    scores: List[float] = []
    for pred_answer, references in zip(pred_answers, all_gold_answers):
        candidate_tokens = _safe_tokenize_for_overlap(pred_answer)
        reference_tokens_list = [_safe_tokenize_for_overlap(reference) for reference in references if isinstance(reference, str)]
        if len(reference_tokens_list) == 0:
            reference_tokens_list = [[]]

        best_score = 0.0
        for reference_tokens in reference_tokens_list:
            if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
                score = 0.0
            else:
                candidate_counts = Counter(candidate_tokens)
                reference_counts = Counter(reference_tokens)
                matches = sum(min(candidate_counts[token], reference_counts[token]) for token in candidate_counts)
                if matches == 0:
                    score = 0.0
                else:
                    precision = matches / len(candidate_tokens)
                    recall = matches / len(reference_tokens)
                    denominator = (alpha * precision) + ((1.0 - alpha) * recall)
                    harmonic = 0.0 if denominator == 0 else (precision * recall) / denominator

                    chunks = _meteor_chunk_count(candidate_tokens, reference_tokens)
                    penalty = gamma * ((chunks / matches) ** beta) if matches > 0 else 0.0
                    score = harmonic * (1.0 - penalty)

            if score > best_score:
                best_score = score

        scores.append(best_score)

    return float(sum(scores) / len(scores))


def compute_supplementary_text_overlap_metrics(
    pred_answers: Sequence[str],
    all_gold_answers: Sequence[Sequence[str]],
) -> Dict[str, float]:
    return {
        "bleu_1": compute_corpus_bleu(pred_answers, all_gold_answers, max_n=1),
        "bleu_2": compute_corpus_bleu(pred_answers, all_gold_answers, max_n=2),
        "bleu_4": compute_corpus_bleu(pred_answers, all_gold_answers, max_n=4),
        "rouge_l": compute_rouge_l(pred_answers, all_gold_answers),
        "meteor": compute_meteor(pred_answers, all_gold_answers),
    }


def derive_answer_type(answer_or_answers: str | Sequence[str]) -> str:
    if isinstance(answer_or_answers, str):
        answer = normalize_answer_for_eval(answer_or_answers)
    elif isinstance(answer_or_answers, Sequence) and len(answer_or_answers) > 0:
        normalized = [normalize_answer_for_eval(item) for item in answer_or_answers if isinstance(item, str)]
        if len(normalized) == 0:
            answer = ""
        else:
            answer = Counter(normalized).most_common(1)[0][0]
    else:
        answer = ""

    if answer in {"yes", "no"}:
        return "yes/no"

    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
    }
    cleaned = answer.replace(",", "").replace(".", "", 1)
    if cleaned.isdigit() or answer in number_words:
        return "number"

    return "other"


def compute_answer_type_breakdown(
    pred_answers: Sequence[str],
    gold_answers_or_annotations: Sequence[Any],
    answer_types: Sequence[str] | None = None,
) -> dict:
    if len(pred_answers) != len(gold_answers_or_annotations):
        raise ValueError(
            "pred_answers and gold_answers_or_annotations length mismatch: "
            f"{len(pred_answers)} vs {len(gold_answers_or_annotations)}"
        )

    valid_types = ["yes/no", "number", "other"]
    counts = {answer_type: 0 for answer_type in valid_types}
    exact_correct = {answer_type: 0 for answer_type in valid_types}
    soft_scores = {answer_type: 0.0 for answer_type in valid_types}

    for index, pred_answer in enumerate(pred_answers):
        record = gold_answers_or_annotations[index]

        provided_type = None
        if answer_types is not None:
            provided_type = answer_types[index]
        elif isinstance(record, dict):
            provided_type = record.get("answer_type")

        if provided_type not in valid_types:
            if isinstance(record, dict):
                answers = record.get("all_gold_answers")
                majority = record.get("gold_majority_answer", "")
                source = answers if isinstance(answers, Sequence) and not isinstance(answers, str) and len(answers) > 0 else majority
                resolved_type = derive_answer_type(source)
            elif isinstance(record, Sequence) and not isinstance(record, str):
                resolved_type = derive_answer_type(record)
            else:
                resolved_type = derive_answer_type(str(record))
        else:
            resolved_type = provided_type

        if resolved_type not in valid_types:
            raise ValueError(f"Unsupported answer type resolved: {resolved_type}")

        counts[resolved_type] += 1

        if isinstance(record, dict):
            all_gold = record.get("all_gold_answers")
            majority = record.get("gold_majority_answer", "")
            if not isinstance(all_gold, Sequence) or isinstance(all_gold, str) or len(all_gold) == 0:
                all_gold = [majority]
            gold_majority = normalize_answer_for_eval(majority)
        elif isinstance(record, Sequence) and not isinstance(record, str):
            all_gold = list(record)
            if len(all_gold) == 0:
                all_gold = [""]
            normalized_all = [normalize_answer_for_eval(item) for item in all_gold]
            gold_majority = Counter(normalized_all).most_common(1)[0][0]
        else:
            all_gold = [str(record)]
            gold_majority = normalize_answer_for_eval(str(record))

        if exact_match_after_normalization(pred_answer, gold_majority):
            exact_correct[resolved_type] += 1

        soft_scores[resolved_type] += compute_vqa_soft_accuracy([pred_answer], [all_gold])

    breakdown = {"overall_count": len(pred_answers), "types": {}}
    for answer_type in valid_types:
        type_count = counts[answer_type]
        exact_acc = (exact_correct[answer_type] / type_count) if type_count > 0 else 0.0
        soft_acc = (soft_scores[answer_type] / type_count) if type_count > 0 else 0.0
        breakdown["types"][answer_type] = {
            "count": type_count,
            "exact_match_accuracy": exact_acc,
            "vqa_soft_accuracy": soft_acc,
        }

    return breakdown


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return compute_top1_accuracy(predictions, labels)


def update_accuracy_counters(
    logits: torch.Tensor,
    labels: torch.Tensor,
    counters: Dict[str, int],
) -> None:
    predictions = logits.argmax(dim=1)
    counters["correct"] += int((predictions == labels).sum().item())
    counters["total"] += int(labels.numel())


def compute_accuracy(counters: Dict[str, int]) -> float:
    total = counters.get("total", 0)
    if total == 0:
        return 0.0
    return counters.get("correct", 0) / total


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_failure_rows(
    question_ids: List[int],
    questions: List[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    answer_vocab: List[str],
    attention_weights: torch.Tensor,
) -> List[dict]:
    rows: List[dict] = []
    for index in range(len(question_ids)):
        gold_idx = int(labels[index].item())
        pred_idx = int(predictions[index].item())
        if gold_idx == pred_idx:
            continue

        sample_attention = attention_weights[index].detach().cpu()
        hops = []
        for hop_index in range(sample_attention.size(0)):
            hop_weights = sample_attention[hop_index]
            top_regions = torch.topk(hop_weights, k=min(5, hop_weights.numel()))
            hops.append(
                {
                    "hop": hop_index,
                    "top_region_indices": top_regions.indices.tolist(),
                    "top_region_weights": [float(v) for v in top_regions.values.tolist()],
                }
            )

        rows.append(
            {
                "question_id": int(question_ids[index]),
                "question": questions[index],
                "gold_answer": answer_vocab[gold_idx],
                "predicted_answer": answer_vocab[pred_idx],
                "attention_debug": hops,
            }
        )
    return rows
