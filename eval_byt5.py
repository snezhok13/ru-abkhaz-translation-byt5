import argparse
import random

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train_byt5 import TASK_PREFIX, load_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--corpus", default="ab-ru-parallel.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.corpus, max_src_chars=256, max_tgt_chars=256)
    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[: args.val_size]

    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions: list[str] = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        sources = [src for src, _ in batch]
        inputs = tokenizer(
            [TASK_PREFIX + src for src in sources],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    scores = [
        sacrebleu.sentence_bleu(prediction, [reference]).score
        for prediction, (_, reference) in zip(predictions, pairs)
    ]
    print(f"examples={len(scores)}")
    print(f"mean_sentence_bleu={sum(scores) / len(scores):.4f}")
    print("\nExamples:")
    for i in range(min(5, len(pairs))):
        print("---")
        print("RU:", pairs[i][0])
        print("REF:", pairs[i][1])
        print("HYP:", predictions[i])


if __name__ == "__main__":
    main()
