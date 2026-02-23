import math
import re

def extract_boxed_answer(solution: str) -> str | None:
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i, ch in enumerate(solution[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i]
    return None

def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.lower()

    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\tfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")

    text = text.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")

    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\leq", "\\le").replace("\\geq", "\\ge")

    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r",and", ",", text)
    text = re.sub(r",or", ",", text)

    text = text.replace("$", "")
    return text

def _strip_var_prefix(text: str) -> str:
    m = re.match(r"^[a-z]=(.+)$", text)
    return m.group(1) if m else text


def _radical_coeff(s: str) -> float:
    if not s or s == "+":
        return 1.0
    if s == "-":
        return -1.0
    return float(s)


def try_parse_number(text: str) -> float | None:
    text = _strip_var_prefix(text.strip())

    m = re.match(r"^(-?\d+)\\frac\{(\d+)\}\{(\d+)\}$", text)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if den == 0:
            return None
        sign = -1 if whole < 0 else 1
        return whole + sign * num / den

    m = re.match(r"^(-?)\\frac\{([^}]+)\}\{([^}]+)\}$", text)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        try:
            return sign * float(m.group(2)) / float(m.group(3))
        except (ValueError, ZeroDivisionError):
            return None

    m = re.match(r"^(-?\d*)?\\sqrt\{(\d+)\}$", text)
    if m:
        return _radical_coeff(m.group(1)) * math.sqrt(int(m.group(2)))

    m = re.match(r"^(-?\d*)\\sqrt\{(\d+)\}([+-][\d.]+)$", text)
    if m:
        return _radical_coeff(m.group(1)) * math.sqrt(int(m.group(2))) + float(m.group(3))

    m = re.match(r"^(-?[\d.]+)([+-]\d*)\\sqrt\{(\d+)\}$", text)
    if m:
        return float(m.group(1)) + _radical_coeff(m.group(2)) * math.sqrt(int(m.group(3)))

    m = re.match(r"^(-?[\d.]+)/(-?[\d.]+)$", text)
    if m:
        try:
            d = float(m.group(2))
            return float(m.group(1)) / d if d != 0 else None
        except ValueError:
            return None

    try:
        return float(text)
    except ValueError:
        return None

class Rewards:

    def answers_match(self, predicted: str, oracle: str) -> bool:
        pred_norm = normalize_answer(predicted)
        oracle_norm = normalize_answer(oracle)

        if pred_norm == oracle_norm:
            return True

        pred_num = try_parse_number(pred_norm)
        oracle_num = try_parse_number(oracle_norm)
        if pred_num is not None and oracle_num is not None:
            return math.isclose(pred_num, oracle_num, rel_tol=1e-6, abs_tol=1e-9)

        return False

    def score_correctness(self,completion: str, oracle_answer: str) -> float:
        match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if match is None:
            return 0.0
        predicted = match.group(1).strip()
        return 1.0 if self.answers_match(predicted, oracle_answer) else 0.0


    def score_format(self, completion: str) -> float:
        score = 0.0
        think_match = re.search(r"<think>(.+?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>(.+?)</answer>", completion, re.DOTALL)
        if think_match and think_match.group(1).strip():
            score += 0.05
        if answer_match and answer_match.group(1).strip():
            score += 0.05
        return score

    def score_reasoning(self, completion: str) -> float:
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

    def score_completion(self, completion: str, oracle_answer: str) -> float:
        correctness = self.score_correctness(completion, oracle_answer)
        fmt = self.score_format(completion)
        reasoning = self.score_reasoning(completion) if correctness >= 1.0 else 0.0
        return correctness + fmt + reasoning
