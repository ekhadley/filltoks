"""Olivia Velte's prompt format, verbatim from orvelte/fillertokensminiproject scripts/prompts.py @ 7f02adc."""
import re

N_FEW_SHOT = 10

# One system prompt for every condition; mentions filler neutrally, never names its type.
SYSTEM = (
    "You will be given a list of variable definitions followed by a question. "
    "Each variable equals either a number or an expression that refers to an "
    "earlier variable (for example 'twice the number for X plus 3'). Resolve "
    "the references to work out the value the question asks for, then answer "
    "immediately with just the number, nothing else. No explanation, no words, "
    "no reasoning, just the number. Some filler text may follow the question "
    "on the 'Filler:' line before the answer."
)


def make_filler(kind, k):
    """k filler units (not tokens). kind: 'dots' or 'counting'."""
    if k == 0:
        return ""
    if kind == "dots":
        return " ".join(["."] * k)
    if kind == "counting":
        return " ".join(str(i) for i in range(1, k + 1))
    raise ValueError(kind)


def user_turn(item, filler):
    defs = "\n".join(f"{name} = {value}" for name, value in item["definitions"])
    # k=0 keeps the bare 'Filler:' label line on purpose (the label is itself a position).
    filler_line = f"Filler: {filler}" if filler else "Filler:"
    return f"{defs}\nQuestion: {item['question']}\n\n{filler_line}\n\nAnswer:"


def build_messages(few_shot, item, kind="dots", k=0, n_shot=N_FEW_SHOT):
    """Chat messages; every few-shot example shows the same filler condition as the test item."""
    filler = make_filler(kind, k)
    messages = [{"role": "system", "content": SYSTEM}]
    for fs in few_shot[:n_shot]:
        messages.append({"role": "user", "content": user_turn(fs, filler)})
        messages.append({"role": "assistant", "content": str(fs["answer"])})
    messages.append({"role": "user", "content": user_turn(item, filler)})
    return messages


_INT = re.compile(r"\s*(-?\d+)\s*")


def parse_answer(text):
    """Strict parse: the whole response must be a single signed integer, else None."""
    m = _INT.fullmatch(text or "")
    return int(m.group(1)) if m else None
