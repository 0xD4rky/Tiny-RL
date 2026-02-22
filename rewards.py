import math
import re


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.lower()
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    text = text.replace("$", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    return text


def try_parse_number(text: str) -> float | None:
    text = text.strip()

    m = re.match(r"^-?\\frac\{([^}]+)\}\{([^}]+)\}$", text)
    if m:
        text = f"{m.group(1)}/{m.group(2)}"

    m = re.match(r"^(-?[\d.]+)/([\d.]+)$", text)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))

    try:
        return float(text)
    except ValueError:
        return None


def answers_match(predicted: str, oracle: str) -> bool:
    pred_norm = normalize_answer(predicted)
    oracle_norm = normalize_answer(oracle)

    if pred_norm == oracle_norm:
        return True

    pred_num = try_parse_number(pred_norm)
    oracle_num = try_parse_number(oracle_norm)
    if pred_num is not None and oracle_num is not None:
        return math.isclose(pred_num, oracle_num, rel_tol=1e-6, abs_tol=1e-9)

    return False


def score_correctness(completion: str, oracle_answer: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
    if match is None:
        return 0.0

    predicted = match.group(1).strip()
    if answers_match(predicted, oracle_answer):
        return 1.0

    if normalize_answer(oracle_answer) in normalize_answer(predicted):
        return 0.5

    return 0.0


def score_format(completion: str) -> float:
    score = 0.0
    if "<think>" in completion and "</think>" in completion:
        score += 0.05
    if "<answer>" in completion and "</answer>" in completion:
        score += 0.05
    return score


def score_reasoning(completion: str) -> float:
    think_match = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
    if not think_match:
        return 0.0

    thinking = think_match.group(1).strip()
    if len(thinking) < 50:
        return 0.0

    words = thinking.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return -0.2

    return 0.1


def score_completion(completion: str, oracle_answer: str) -> float:
    return (
        score_correctness(completion, oracle_answer)
        + score_format(completion)
        + score_reasoning(completion)
    )
