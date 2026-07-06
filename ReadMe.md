# PrefSCom: Preference-Optimized Fine-Tuning for Faithful Smart Contract Comment Generation

This repository contains the source code, datasets, and experimental scripts for the paper:

**PrefSCom: Preference-Optimized Fine-Tuning for Faithful Smart Contract Comment Generation**

PrefSCom is designed for **Solidity  comment generation**. Given a Solidity function, PrefSCom generates a concise, readable, and semantically faithful natural-language comment that describes explicit function semantics, such as function operations, access-control constraints, state updates, conditions, and return values.

------

## Approach

https://github.com/shawnwbts/SIRCOT-main/blob/master/SIRCOT.png

------

## Repository Structure

The repository is organized as follows:

```text
PrefSCom
│
├── dataset/
│   ├── dpo_pool.csv
│   ├── dpo_pool.jsonl
│   ├── sft_train.csv
│   ├── sft_train.jsonl
│   ├── test.csv
│   ├── test.jsonl
│   ├── validation.csv
│   └── validation.jsonl
│
├── evaluation/
│   ├── S-CSemSimEvaluation.py
│   ├── S-SSemSimEvaluation.py
│   └── S-STexSimEvaluation.py
│
├── build_dpo_pairs.py
├── generate_dpo_candidates.py
├── score_dpo_candidates.py
├── train_sft_model.py
├── train_dpo_model.py
├── save_predictions.py
├── pic.png
└── README.md
```

------

The main files are described below.

| File or Folder               | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `dataset/`                   | Processed Solidity function-comment datasets for SFT, validation, testing, and DPO preference construction. |
| `evaluation/`                | Scripts for automatic evaluation, including textual similarity, summary-summary semantic similarity, and summary-code semantic similarity. |
| `train_sft_model.py`         | Performs supervised fine-tuning on Solidity function-comment pairs. |
| `generate_dpo_candidates.py` | Generates candidate comments for each function in the DPO pool. |
| `score_dpo_candidates.py`    | Scores candidate comments using the weighted preference function. |
| `build_dpo_pairs.py`         | Constructs DPO preference pairs from scored candidates.      |
| `train_dpo_model.py`         | Performs DPO-based preference optimization.                  |
| `save_predictions.py`        | Generates comments using the trained SFT or DPO model.       |
| `pic.png`                    | Overview figure of PrefSCom.                                 |

## How to Run PrefSCom

### Step 1: Supervised Fine-Tuning

------

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_sft_model.py \
  --model_name_or_path /xx/Qwen2.5-Coder-7B-Instruct/ \
  --train_file dataset/sft_train.jsonl \
  --validation_file dataset/validation.jsonl \
  --output_dir outputs/prefscom_sft_qwen2_5_coder_7b \
  --use_qlora \
  --bf16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --max_length 1024 \
  --response_max_length 64 \
  --eval_steps 200 \
  --save_steps 200
```



### Step 2: Generate SFT Predictions

After SFT, generate comments on the test set.

```bash
CUDA_VISIBLE_DEVICES=1 \
python save_predictions.py \
  --base_model_path /xx/Qwen2.5-Coder-7B-Instruct/ \
  --sft_adapter_path outputs/prefscom_sft_qwen2_5_coder_7b \
  --test_file dataset/test.jsonl \
  --output_file outputs/sft_predictions/test_predictions_sft.jsonl \
  --load_in_4bit \
  --bf16 \
  --max_new_tokens 32
```

------

### Step 3: Generate DPO Candidate Comments

Generate candidate comments for each function in the DPO pool.

```bash
CUDA_VISIBLE_DEVICES=1 \
python generate_dpo_candidates.py \
  --base_model_path /xx/Qwen2.5-Coder-7B-Instruct/ \
  --sft_adapter_path outputs/prefscom_sft_qwen2_5_coder_7b \
  --dpo_pool_file dataset/dpo_pool.jsonl \
  --output_file dataset/candidate_comments.jsonl \
  --load_in_4bit \
  --bf16 \
  --batch_size 2 \
  --max_new_tokens 32
```

------

### Step 4: Score DPO Candidates

Score the generated candidate comments. 

```bash
python score_dpo_candidates.py \
  --candidate_file dataset/candidate_comments.jsonl \
  --output_file dataset/scored_candidates.jsonl \
  --side_device cuda:1 \
  --sem_device cuda:1 \
  --side_batch_size 32 \
  --sem_batch_size 32 \
  --w_code 0.45 \
  --w_sem 0.25 \
  --w_sec 0.15 \
  --w_style 0.10 \
  --w_hall 0.25
```

------

### Step 5: Build DPO Preference Pairs

Construct preference pairs from scored candidates.

```bash
python build_dpo_pairs.py \
  --scored_file dataset/scored_candidates.jsonl \
  --output_file dataset/dpo_pairs.jsonl \
  --min_margin 0.20 \
  --max_pairs_per_sample 3
```

------

### Step 6: Train the DPO Model

Train PrefSCom with DPO-based preference optimization.

```bash
CUDA_VISIBLE_DEVICES=1 \
python train_dpo_model.py \
  --base_model_path /xx/Qwen2.5-Coder-7B-Instruct/ \
  --sft_adapter_path outputs/prefscom_sft_qwen2_5_coder_7b \
  --dpo_file dataset/dpo_pairs.jsonl \
  --output_dir outputs/prefscom \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 1152 \
  --max_prompt_length 1024 \
  --max_response_length 64 \
  --beta 0.03 \
  --average_log_prob \
  --sft_loss_weight 0.10 \
  --num_train_epochs 1 \
  --learning_rate 5e-7 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --ref_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_ratio 0.1 \
  --logging_steps 10 \
  --eval_steps 50 \
  --save_steps 100 \
  --overwrite_ref_cache
```

The main DPO hyperparameters include:

| Hyperparameter           | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `--beta`                 | Controls the strength of DPO optimization.                   |
| `--sft_loss_weight`      | Controls supervised regularization during DPO training.      |
| `--min_margin`           | Controls the confidence threshold for preference-pair construction. |
| `--max_pairs_per_sample` | Limits the number of DPO pairs per function.                 |

### Step 7: Generate PrefSCom Predictions

After DPO training, generate final PrefSCom comments on the test set.

```bash
CUDA_VISIBLE_DEVICES=1 \
python save_predictions.py \
  --base_model_path /xx/Qwen2.5-Coder-7B-Instruct/ \
  --sft_adapter_path outputs/prefscom/best \
  --test_file dataset/test.jsonl \
  --output_file outputs/prefscom/test_predictions_prefscom.jsonl \
  --load_in_4bit \
  --bf16 \
  --max_new_tokens 24
```

------

## License

This project is released for academic research purposes. Please refer to the license file for more details.

------

## Contact

If you have any questions, please open an issue or contact the authors.