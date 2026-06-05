# RU → Abkhazian Translation (ByT5)

Fine-tuning `google/byt5-small` для машинного перевода русский → абхазский.

## Соревнование
**Yandex ML Тренировка — Data Dojo 114**

## Результат
- Итоговый score: **32,44**
- Пайплайн: обучение → оценка BLEU → экспорт весов → submission

## Стек
Python, PyTorch, Hugging Face Transformers, sacrebleu, Google Colab (GPU)

## Запуск
```bash
pip install torch transformers sacrebleu
python train_byt5.py --corpus ab-ru-parallel.csv --output_dir weights --epochs 1
python eval_byt5.py --weights weights --corpus ab-ru-parallel.csv
```

Корпус `ab-ru-parallel.csv` не включён из-за размера — используйте свой файл или скачайте из соревнования.
