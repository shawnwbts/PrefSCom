
"""
PrefSCom: Save SFT Baseline Predictions

功能：
1. 加载本地 Qwen2.5-Coder-7B-Instruct base model
2. 加载已经训练好的 SFT LoRA/QLoRA adapter
3. 在 data/sft/test.jsonl 上生成注释
4. 保存 SFT only baseline 的预测结果

输入：
    data/sft/test.jsonl

输出：
    outputs/sft_predictions/test_predictions_sft.jsonl

输出格式：
{
  "id": "...",
  "instruction": "...",
  "function": "...",
  "reference": "...",
  "prediction": "..."
}

CUDA_VISIBLE_DEVICES=0 \
python save_sft_baseline_predictions.py \
  --base_model_path /data/wb/models/Qwen2.5-Coder-7B-Instruct/ \
  --sft_adapter_path outputs/prefscom_sft_qwen2_5_coder_7b_prompt_v2 \
  --test_file data/sft/test.jsonl \
  --output_file outputs/sft_predictions/test_predictions_sft_prompt_v2.jsonl \
  --load_in_4bit \
  --bf16 \
  --max_new_tokens 32
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)

from peft import PeftModel


# ============================================================
# 1. Data Loading
# ============================================================

def load_jsonl(data_path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    samples = []

    with open(data_path, "r", encoding=encoding, errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error at line {line_no}: {e}")
                continue

            if not all(k in item for k in ["instruction", "input", "output"]):
                print(f"[Warning] Missing fields at line {line_no}: {item.keys()}")
                continue

            samples.append({
                "id": item.get("id", f"sample_{line_no}"),
                "instruction": str(item["instruction"]).strip(),
                "function": str(item["input"]).strip(),
                "reference": str(item["output"]).strip(),
            })

    print(f"[Data] Loaded {len(samples)} samples from {data_path}")
    return samples


# ============================================================
# 2. Prompt Construction
# ============================================================

def build_prompt(tokenizer, instruction: str, function_code: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Solidity developer. "
                "Generate one short human-written style comment for the given Solidity function. "
                "The comment should be concise, natural, and faithful to the function behavior. "
                "Use 5 to 15 words when possible. "
                "Do not add explanations, return descriptions, ownership claims, emergency behavior, "
                "or security guarantees unless they are explicitly shown in the function."
            )
        },
        {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"Solidity function:\n"
                f"{function_code}\n\n"
                "Generate exactly one short comment. "
                "Do not include code, markdown, or additional explanation."
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


# ============================================================
# 3. Model Loading
# ============================================================

def load_sft_model(args):
    print(f"[Tokenizer] Loading tokenizer from: {args.base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if args.load_in_4bit:
        print("[Model] Loading base model in 4-bit mode.")

        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    print(f"[Model] Loading base model from: {args.base_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )

    print(f"[PEFT] Loading SFT adapter from: {args.sft_adapter_path}")

    model = PeftModel.from_pretrained(
        model,
        args.sft_adapter_path,
        is_trainable=False,
    )

    model.eval()

    return model, tokenizer


# ============================================================
# 4. Post-processing
# ============================================================

def clean_prediction(text: str) -> str:
    """
    清理模型输出，尽量得到单句注释。
    """
    if text is None:
        return ""

    text = text.strip()

    # 去掉可能残留的角色标记
    bad_prefixes = [
        "assistant",
        "Assistant:",
        "Comment:",
        "### Comment:",
        "Here is the comment:",
        "The comment is:",
    ]

    for p in bad_prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    # 如果模型输出了多行，只取第一段非空内容
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        text = lines[0]

    # 去掉 markdown 代码块标记
    text = text.replace("```solidity", "").replace("```", "").strip()

    # 去掉多余引号
    if len(text) >= 2 and text[0] in ['"', "'"] and text[-1] == text[0]:
        text = text[1:-1].strip()

    return text


# ============================================================
# 5. Generation
# ============================================================

def generate_one(model, tokenizer, sample: Dict[str, Any], args) -> str:
    prompt = build_prompt(
        tokenizer=tokenizer,
        instruction=sample["instruction"],
        function_code=sample["function"]
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": args.repetition_penalty,
    }

    # 删除 None，避免部分 transformers 版本报警
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_ids = outputs[0][input_len:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

    prediction = clean_prediction(prediction)

    return prediction


def save_predictions(model, tokenizer, samples: List[Dict[str, Any]], args):
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 支持断点续跑：如果输出文件已存在，跳过已完成 id
    done_ids = set()

    if output_path.exists() and args.resume:
        print(f"[Resume] Found existing output file: {output_path}")

        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done_ids.add(item["id"])
                except Exception:
                    continue

        print(f"[Resume] Loaded {len(done_ids)} completed samples.")

    mode = "a" if args.resume else "w"

    count = 0

    with open(output_path, mode, encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="Generating SFT predictions"):
            if sample["id"] in done_ids:
                continue

            try:
                prediction = generate_one(model, tokenizer, sample, args)
            except RuntimeError as e:
                print(f"\n[Error] RuntimeError on sample {sample['id']}: {e}")
                prediction = ""

                if torch.cuda.is_available:
                    torch.cuda.empty_cache()

            record = {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "function": sample["function"],
                "reference": sample["reference"],
                "prediction": prediction,
                "model": "sft",
                "base_model_path": args.base_model_path,
                "sft_adapter_path": args.sft_adapter_path,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            count += 1

            if args.max_samples is not None and count >= args.max_samples:
                print(f"[Info] Reached max_samples={args.max_samples}. Stop.")
                break

    print(f"[Done] Predictions saved to: {output_path}")


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/data/wb/models/Qwen2.5-Coder-7B-Instruct/",
        help="本地 base model 路径。"
    )

    parser.add_argument(
        "--sft_adapter_path",
        type=str,
        default="outputs/prefscom_sft_qwen2_5_coder_7b",
        help="SFT LoRA/QLoRA adapter 路径。"
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="data/sft/test.jsonl",
        help="测试集 JSONL 文件。"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/sft_predictions/test_predictions_sft.jsonl",
        help="预测结果保存路径。"
    )

    parser.add_argument("--data_encoding", type=str, default="utf-8")

    # generation config
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # model loading config
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # running config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("PrefSCom: Save SFT Baseline Predictions")
    print("=" * 80)
    print(f"Base model      : {args.base_model_path}")
    print(f"SFT adapter     : {args.sft_adapter_path}")
    print(f"Test file       : {args.test_file}")
    print(f"Output file     : {args.output_file}")
    print(f"Load in 4bit    : {args.load_in_4bit}")
    print("=" * 80)

    samples = load_jsonl(
        data_path=args.test_file,
        encoding=args.data_encoding,
    )

    if args.max_samples is not None:
        print(f"[Info] max_samples is set to {args.max_samples}")

    model, tokenizer = load_sft_model(args)

    save_predictions(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        args=args,
    )


if __name__ == "__main__":
    main()