
"""
PrefSCom: Train SFT Model

功能：
1. 读取 data/sft/sft_train.jsonl 和 data/sft/validation.jsonl
2. 使用 Qwen2.5-Coder-7B-Instruct 作为 base model
3. 使用 LoRA / QLoRA 进行监督微调
4. 训练智能合约函数注释生成模型
5. 保存 SFT adapter，供后续 DPO 阶段使用

数据格式：
{
  "id": "sft_train_xxx",
  "instruction": "Generate a concise and accurate comment for the given Solidity function.",
  "input": "function ...",
  "output": "..."
}
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 1. Dataset
# ============================================================

class SmartContractCommentSFTDataset(Dataset):
    """
    用于智能合约注释生成的 SFT Dataset。

    训练目标：
    给定 Solidity function，生成简洁、准确、代码忠实的注释。

    注意：
    labels 中 prompt 部分设置为 -100，只在 assistant response 上计算 loss。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        response_max_length: int = 128,
        encoding: str = "utf-8",
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_max_length = response_max_length
        self.encoding = encoding

        self.samples = self._load_jsonl(data_path)
        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")

    def _load_jsonl(self, data_path: str) -> List[Dict[str, Any]]:
        samples = []

        with open(data_path, "r", encoding=self.encoding, errors="replace") as f:
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

                instruction = str(item["instruction"]).strip()
                function_code = str(item["input"]).strip()
                comment = str(item["output"]).strip()

                if not instruction or not function_code or not comment:
                    continue

                samples.append({
                    "id": item.get("id", f"sample_{line_no}"),
                    "instruction": instruction,
                    "function": function_code,
                    "comment": comment,
                })

        return samples


    def build_prompt(self, instruction: str, function_code: str) -> str:
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

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.samples[idx]

        prompt_text = self.build_prompt(
            instruction=item["instruction"],
            function_code=item["function"]
        )

        response_text = item["comment"].strip()

        # Qwen chat template 已经生成了 assistant 开头，这里 response 后加 eos
        if self.tokenizer.eos_token is not None:
            response_text = response_text + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        response_ids = self.tokenizer(
            response_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.response_max_length,
        )["input_ids"]

        # 如果 prompt + response 超过 max_length，优先保留 response，截断 prompt 左侧
        total_length = len(prompt_ids) + len(response_ids)

        if total_length > self.max_length:
            keep_prompt_len = self.max_length - len(response_ids)

            if keep_prompt_len <= 0:
                # 极端情况：response 本身太长，截断 response
                response_ids = response_ids[:self.max_length]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-keep_prompt_len:]

        input_ids = prompt_ids + response_ids
        attention_mask = [1] * len(input_ids)

        # prompt 部分不计算 loss；response 部分计算 loss
        labels = [-100] * len(prompt_ids) + response_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# 2. Data Collator
# ============================================================

class SFTDataCollator:
    """
    动态 padding。
    labels 的 padding 使用 -100。
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)

        if self.pad_to_multiple_of is not None:
            max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        pad_token_id = self.tokenizer.pad_token_id

        for x in features:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            labels = x["labels"]

            pad_len = max_len - len(input_ids)

            input_ids_batch.append(input_ids + [pad_token_id] * pad_len)
            attention_mask_batch.append(attention_mask + [0] * pad_len)
            labels_batch.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


# ============================================================
# 3. Model Loading
# ============================================================

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = 100 * trainable_params / all_params

    print(
        f"[Model] Trainable params: {trainable_params:,} | "
        f"All params: {all_params:,} | "
        f"Trainable ratio: {ratio:.4f}%"
    )


def load_model_and_tokenizer(args):
    print(f"[Tokenizer] Loading tokenizer from: {args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # 量化配置
    if args.use_qlora:
        print("[Model] Using QLoRA 4-bit quantization.")

        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    print(f"[Model] Loading model from: {args.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )

    model.config.use_cache = False

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # 开启 gradient checkpointing 时建议关闭 use_cache
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]

    print(f"[LoRA] Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    return model, tokenizer


# ============================================================
# 4. Generation Test
# ============================================================

def quick_generation_test(model, tokenizer, output_dir: str):
    """
    简单测试训练后的模型是否能生成注释。
    """

    print("\n[Quick Test] Running generation test...")

    function_code = (
        "function withdraw(uint256 amount) public { "
        "require(block.timestamp >= unlockTime[msg.sender]); "
        "payable(msg.sender).transfer(amount); "
        "}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Solidity developer. "
                "Generate concise, accurate, code-faithful, and security-aware "
                "comments for smart contract functions."
            )
        },
        {
            "role": "user",
            "content": (
                "Generate a concise and accurate comment for the given Solidity function.\n\n"
                f"Solidity function:\n{function_code}\n\n"
                "Please generate only one concise comment."
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n[Quick Test] Function:")
    print(function_code)
    print("\n[Quick Test] Output:")
    print(generated)

    save_path = Path(output_dir) / "quick_generation_test.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Function:\n")
        f.write(function_code + "\n\n")
        f.write("Generated:\n")
        f.write(generated + "\n")

    print(f"[Quick Test] Saved to: {save_path}")


# ============================================================
# 5. TrainingArguments compatibility
# ============================================================

def build_training_args(args):
    """
    不同 transformers 版本中，evaluation_strategy / eval_strategy 可能不同。
    这里优先使用 evaluation_strategy，如果版本不兼容，再提示用户修改。
    """

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,

        logging_steps=args.logging_steps,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        bf16=args.bf16,
        fp16=args.fp16,

        gradient_checkpointing=args.gradient_checkpointing,

        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",

        report_to=args.report_to,
        run_name=args.run_name,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    return training_args


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/wb/models/Qwen2.5-Coder-7B-Instruct/",
        help="本地模型路径或 HuggingFace 模型名。"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/sft/sft_train.jsonl",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="data/sft/validation.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/prefscom_sft_qwen2_5_coder_7b",
    )

    # 数据参数
    parser.add_argument("--data_encoding", type=str, default="utf-8")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--response_max_length", type=int, default=128)

    # 训练参数
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # LoRA / QLoRA
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # 精度参数
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default="PrefSCom-SFT")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--quick_test", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PrefSCom: SFT Model Training")
    print("=" * 80)
    print(f"Base model       : {args.model_name_or_path}")
    print(f"Train file       : {args.train_file}")
    print(f"Validation file  : {args.validation_file}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Use QLoRA        : {args.use_qlora}")
    print(f"bf16             : {args.bf16}")
    print(f"fp16             : {args.fp16}")
    print("=" * 80)

    # 1. 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # 2. 加载数据
    train_dataset = SmartContractCommentSFTDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        response_max_length=args.response_max_length,
        encoding=args.data_encoding,
    )

    eval_dataset = SmartContractCommentSFTDataset(
        data_path=args.validation_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        response_max_length=args.response_max_length,
        encoding=args.data_encoding,
    )

    data_collator = SFTDataCollator(tokenizer)

    # 3. 训练参数
    training_args = build_training_args(args)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 5. 开始训练
    print("\n[Training] Start training...")
    train_result = trainer.train()

    # 6. 保存最终模型
    print("\n[Saving] Saving final LoRA adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7. 保存训练指标
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)

    train_metrics_path = Path(args.output_dir) / "train_metrics.json"
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2, ensure_ascii=False)

    print(f"[Saving] Train metrics saved to: {train_metrics_path}")

    # 8. 最终验证
    print("\n[Evaluation] Running final evaluation...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)

    eval_metrics_path = Path(args.output_dir) / "eval_metrics.json"
    with open(eval_metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    print(f"[Saving] Eval metrics saved to: {eval_metrics_path}")
    print(eval_metrics)

    # 9. 简单生成测试
    if args.quick_test:
        quick_generation_test(model, tokenizer, args.output_dir)

    print("\n" + "=" * 80)
    print("SFT training completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()