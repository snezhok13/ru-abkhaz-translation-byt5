# Colab cells for Russian -> Abkhazian training

Upload these files to the Colab working directory:

- `ab-ru-parallel.csv`
- `train_byt5.py`
- `eval_byt5.py`

Recommended runtime:

- `Runtime -> Change runtime type -> T4 GPU`

## 1. Check GPU

```python
!nvidia-smi
```

## 2. Install dependencies

```python
!pip install -U torch transformers accelerate sentencepiece safetensors sacrebleu
```

## 3. Upload files manually

```python
from google.colab import files
uploaded = files.upload()
```

Upload:

- `ab-ru-parallel.csv`
- `train_byt5.py`
- `eval_byt5.py`

## 4. Quick first training run

This is the first practical run. It trains on 20k examples for 1 epoch.

```python
!python train_byt5.py \
  --model_name google/byt5-small \
  --corpus ab-ru-parallel.csv \
  --output_dir weights \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum 8 \
  --learning_rate 5e-4 \
  --max_train_samples 20000 \
  --max_source_length 256 \
  --max_target_length 256
```

## 5. Evaluate

```python
!python eval_byt5.py --weights weights --corpus ab-ru-parallel.csv --val_size 1000
```

## 6. If BLEU is promising, train longer

```python
!rm -rf weights
!python train_byt5.py \
  --model_name google/byt5-small \
  --corpus ab-ru-parallel.csv \
  --output_dir weights \
  --epochs 2 \
  --batch_size 4 \
  --grad_accum 8 \
  --learning_rate 5e-4 \
  --max_train_samples 60000 \
  --max_source_length 256 \
  --max_target_length 256
```

## 7. Zip weights for download

```python
!zip -r weights.zip weights
files.download("weights.zip")
```

After downloading `weights.zip`, unzip it locally into:

`C:\Users\Snezhko\Desktop\соревн.ч.2\weights`

Then build the final contest archive locally:

```powershell
cd "C:\Users\Snezhko\Desktop\соревн.ч.2"
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_submission.ps1
```
