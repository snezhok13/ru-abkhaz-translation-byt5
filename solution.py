import csv
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


INPUT_PATH = "input.pickle"
CORPUS_PATH = "ab-ru-parallel.csv"
MODEL_DIR = "weights"
OUTPUT_PATHS = ("output.json", os.path.join("out", "output.json"))
BATCH_SIZE = 16
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 256
TASK_PREFIX = "translate Russian to Abkhazian: "

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def canonical(text: str) -> str:
    return " ".join(tokenize(text))


def load_memory() -> tuple[dict[str, str], dict[str, str]]:
    exact_votes: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_votes: dict[str, Counter[str]] = defaultdict(Counter)

    with open(CORPUS_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ru = (row.get("ru") or "").strip()
            ab = (row.get("ab") or "").strip()
            if not ru or not ab:
                continue

            norm_ru = normalize(ru)
            exact_votes[norm_ru][ab] += 1
            canon_ru = canonical(ru)
            if canon_ru:
                canonical_votes[canon_ru][ab] += 1

    exact = {ru: votes.most_common(1)[0][0] for ru, votes in exact_votes.items()}
    canon = {ru: votes.most_common(1)[0][0] for ru, votes in canonical_votes.items()}
    return exact, canon


def memory_translate(src: str, exact: dict[str, str], canon: dict[str, str]) -> str | None:
    norm_src = normalize(src)
    if norm_src in exact:
        return exact[norm_src]

    canon_src = canonical(src)
    if canon_src in canon:
        return canon[canon_src]
    return None


def load_model():
    if not os.path.isdir(MODEL_DIR):
        return None, None, None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def model_translate(texts: list[str], tokenizer, model, device: str) -> list[str]:
    if tokenizer is None or model is None:
        return texts

    results: list[str] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            [TASK_PREFIX + text for text in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )

        results.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return [text.strip() for text in results]


def main() -> None:
    exact, canon = load_memory()
    tokenizer, model, device = load_model()

    with open(INPUT_PATH, "rb") as f:
        rows = pickle.load(f)

    results = []
    model_jobs: list[tuple[int, str]] = []
    for idx, row in enumerate(rows):
        src = row.get("src", "")
        translation = memory_translate(src, exact, canon)
        results.append({"rid": row["rid"], "translation": translation or ""})
        if translation is None:
            model_jobs.append((idx, src))

    if model_jobs:
        model_outputs = model_translate([src for _, src in model_jobs], tokenizer, model, device)
        for (idx, src), translation in zip(model_jobs, model_outputs):
            results[idx]["translation"] = translation or src

    for path in OUTPUT_PATHS:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
