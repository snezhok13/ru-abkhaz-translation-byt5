import argparse
import csv
import os
import random
import re

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


SPACE_RE = re.compile(r"\s+")
TASK_PREFIX = "translate Russian to Abkhazian: "


def clean(text: str) -> str:
    text = (text or "").replace("\ufeff", " ")
    return SPACE_RE.sub(" ", text).strip()


def canonical_ru(text: str) -> str:
    text = clean(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def load_pairs(path: str, max_src_chars: int, max_tgt_chars: int) -> list[tuple[str, str]]:
    best_by_ru: dict[str, tuple[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ru = clean(row.get("ru", ""))
            ab = clean(row.get("ab", ""))
            if not ru or not ab:
                continue
            if len(ru) > max_src_chars or len(ab) > max_tgt_chars:
                continue
            key = canonical_ru(ru)
            if not key:
                continue
            old = best_by_ru.get(key)
            if old is None or len(ab) < len(old[1]):
                best_by_ru[key] = (ru, ab)
    return list(best_by_ru.values())


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_source_length: int, max_target_length: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        src, tgt = self.pairs[idx]
        model_inputs = self.tokenizer(
            TASK_PREFIX + src,
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=tgt,
            max_length=self.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/byt5-small")
    parser.add_argument("--corpus", default="ab-ru-parallel.csv")
    parser.add_argument("--output_dir", default="weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--max_src_chars", type=int, default=256)
    parser.add_argument("--max_tgt_chars", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pairs = load_pairs(args.corpus, args.max_src_chars, args.max_tgt_chars)
    random.shuffle(pairs)
    if args.max_train_samples > 0:
        pairs = pairs[: args.max_train_samples + args.val_size]

    val_size = min(args.val_size, max(1, len(pairs) // 20))
    eval_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    print(f"Loaded pairs: {len(pairs)}")
    print(f"Train pairs:  {len(train_pairs)}")
    print(f"Eval pairs:   {len(eval_pairs)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    train_dataset = TranslationDataset(train_pairs, tokenizer, args.max_source_length, args.max_target_length)
    eval_dataset = TranslationDataset(eval_pairs, tokenizer, args.max_source_length, args.max_target_length)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, "_checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=25,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
