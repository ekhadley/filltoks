"""Olivia Velte's prompt format (orvelte/fillertokensminiproject scripts/prompts.py @ 7f02adc), extended to Brauer et al.'s 1-fact addition items."""
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

# Brauer et al.'s 1-fact addition ("What is the atomic number of iron plus 42?"), with the same neutral filler sentence.
SYSTEM_FACT = (
    "You will be given a question whose answer is a number (for example 'What is "
    "the atomic number of iron plus 42?'). Work out the answer, then answer "
    "immediately with just the number, nothing else. No explanation, no words, "
    "no reasoning, just the number. Some filler text may follow the question "
    "on the 'Filler:' line before the answer."
)
SYSTEM_BY_TYPE = {"chained_var_binding": SYSTEM, "1fact_addition": SYSTEM_FACT, "2fact_addition": SYSTEM_FACT, "nfact_addition": SYSTEM_FACT}


def make_filler(kind, k):
    """k filler units (not tokens). kind: 'dots' or 'counting'; 'wiki' text is tokenizer-dependent and comes from utils.wiki_filler."""
    if k == 0:
        return ""
    if kind == "dots":
        return " ".join(["."] * k)
    if kind == "counting":
        return " ".join(str(i) for i in range(1, k + 1))
    raise ValueError(kind)


def user_turn(item, filler, before=""):
    """before: a second filler placed above the definitions (Olivia's 'before everything' placement); the 'Filler:' line before 'Answer:' stays, bare when filler is empty."""
    lines = [f"{name} = {value}" for name, value in item.get("definitions", [])] + [f"Question: {item['question']}"]
    # k=0 keeps the bare 'Filler:' label line on purpose (the label is itself a position).
    filler_line = f"Filler: {filler}" if filler else "Filler:"
    return (f"Filler: {before}\n\n" if before else "") + "\n".join(lines) + f"\n\n{filler_line}\n\nAnswer:"


def build_messages(few_shot, item, kind="dots", k=0, n_shot=N_FEW_SHOT, fillers=None, befores=None):
    """Chat messages; every few-shot example shows the same filler condition and placement as the test item. Given filler strings (one per shot, then the item's) override kind and k; befores are the fillers above the definitions."""
    fillers = [make_filler(kind, k)] * (n_shot + 1) if fillers is None else fillers
    befores = [""] * (n_shot + 1) if befores is None else befores
    messages = [{"role": "system", "content": SYSTEM_BY_TYPE[item["type"]]}]
    for fs, filler, before in zip(few_shot[:n_shot], fillers, befores):
        messages.append({"role": "user", "content": user_turn(fs, filler, before)})
        messages.append({"role": "assistant", "content": str(fs["answer"])})
    messages.append({"role": "user", "content": user_turn(item, fillers[-1], befores[-1])})
    return messages


_INT = re.compile(r"\s*(-?\d+)\s*")


def parse_answer(text):
    """Strict parse: the whole response must be a single signed integer, else None."""
    m = _INT.fullmatch(text or "")
    return int(m.group(1)) if m else None
