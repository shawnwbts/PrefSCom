
"""
PrefSCom: Train DPO Model

功能：
1. 读取 data/dpo/dpo_pairs.jsonl
2. 加载 base model: Qwen2.5-Coder-7B-Instruct
3. 加载 SFT LoRA adapter，作为 DPO policy 初始模型
4. 预计算 reference logprobs：reference model = 初始 SFT model
5. 使用 DPO loss 继续训练 LoRA adapter
5.1 可选：加入 chosen response 的 SFT/CE 正则项，避免 DPO 偏离 reference-style 注释分布
6. 保存 DPO adapter 到 outputs/prefscom_dpo_qwen2_5_coder_7b/

DPO 数据格式：
{
  "id": "...",
  "prompt": "...",
  "system_prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "chosen_score": ...,
  "rejected_score": ...,
  "margin": ...
}
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler,
    set_seed,
)

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)


# ============================================================
# 1. Dataset
# ============================================================

class DPOPairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, Any]]):
        self.samples = pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error at line {line_no}: {e}")
                continue

            required = ["id", "prompt", "chosen", "rejected"]
            if not all(k in item for k in required):
                print(f"[Warning] Missing required fields at line {line_no}: {item.keys()}")
                continue

            item["prompt"] = str(item["prompt"]).strip()
            item["chosen"] = str(item["chosen"]).strip()
            item["rejected"] = str(item["rejected"]).strip()
            item["system_prompt"] = str(item.get(
                "system_prompt",
                "You are an expert Solidity developer. "
    "Generate one short human-written style comment for the given Solidity function. "
    "The comment should be concise, natural, and faithful to the function behavior. "
    "Use 5 to 15 words when possible. "
    "Do not add explanations, return descriptions, ownership claims, emergency behavior, "
    "or security guarantees unless they are explicitly shown in the function."
            )).strip()

            if not item["prompt"] or not item["chosen"] or not item["rejected"]:
                continue

            data.append(item)

    print(f"[Data] Loaded {len(data)} DPO pairs from {path}")
    return data


def split_train_eval(
    data: List[Dict[str, Any]],
    eval_ratio: float,
    seed: int
):
    if eval_ratio <= 0:
        return data, []

    rng = random.Random(seed)
    data = data[:]
    rng.shuffle(data)

    eval_size = int(len(data) * eval_ratio)
    eval_data = data[:eval_size]
    train_data = data[eval_size:]

    print(f"[Data] Train pairs: {len(train_data)}")
    print(f"[Data] Eval pairs : {len(eval_data)}")

    return train_data, eval_data


# ============================================================
# 2. Prompt and Encoding
# ============================================================

def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def encode_prompt_response(
    tokenizer,
    system_prompt: str,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int,
    max_response_length: int,
):
    """
    编码 prompt + response，并且只在 response 上计算 logprob/loss。

    labels:
      prompt token -> -100
      response token -> token id
    """
    prompt_text = build_chat_prompt(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=prompt
    )

    response_text = response.strip()
    if tokenizer.eos_token is not None:
        response_text = response_text + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )["input_ids"]

    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_response_length,
    )["input_ids"]

    total_len = len(prompt_ids) + len(response_ids)

    if total_len > max_length:
        keep_prompt_len = max_length - len(response_ids)

        if keep_prompt_len <= 0:
            response_ids = response_ids[:max_length]
            prompt_ids = []
        else:
            # 左侧截断 prompt，保留靠近生成位置的函数上下文
            prompt_ids = prompt_ids[-keep_prompt_len:]

    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DPODataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 1152,
        max_prompt_length: int = 1024,
        max_response_length: int = 128,
        pad_to_multiple_of: int = 8,
        include_ref_logps: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.include_ref_logps = include_ref_logps

    def _pad_batch(self, encoded_list: List[Dict[str, List[int]]]):
        max_len = max(len(x["input_ids"]) for x in encoded_list)

        if self.pad_to_multiple_of is not None:
            max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        pad_token_id = self.tokenizer.pad_token_id

        for x in encoded_list:
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

    def __call__(self, features: List[Dict[str, Any]]):
        chosen_encoded = []
        rejected_encoded = []

        ids = []

        for item in features:
            ids.append(item["id"])

            chosen_encoded.append(
                encode_prompt_response(
                    tokenizer=self.tokenizer,
                    system_prompt=item.get("system_prompt", ""),
                    prompt=item["prompt"],
                    response=item["chosen"],
                    max_length=self.max_length,
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length,
                )
            )

            rejected_encoded.append(
                encode_prompt_response(
                    tokenizer=self.tokenizer,
                    system_prompt=item.get("system_prompt", ""),
                    prompt=item["prompt"],
                    response=item["rejected"],
                    max_length=self.max_length,
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length,
                )
            )

        batch = {
            "ids": ids,
            "chosen": self._pad_batch(chosen_encoded),
            "rejected": self._pad_batch(rejected_encoded),
        }

        if self.include_ref_logps:
            batch["ref_chosen_logps"] = torch.tensor(
                [float(x["ref_chosen_logp"]) for x in features],
                dtype=torch.float32
            )
            batch["ref_rejected_logps"] = torch.tensor(
                [float(x["ref_rejected_logp"]) for x in features],
                dtype=torch.float32
            )

        return batch


# ============================================================
# 3. Model Loading
# ============================================================

def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_policy_model(args):
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

    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(f"[PEFT] Loading SFT adapter from: {args.sft_adapter_path}")

    model = PeftModel.from_pretrained(
        model,
        args.sft_adapter_path,
        is_trainable=True,
    )

    model.train()

    print_trainable_parameters(model)

    return model


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(
        f"[Model] Trainable params: {trainable:,} | "
        f"Total params: {total:,} | "
        f"Trainable ratio: {100 * trainable / total:.4f}%"
    )


# ============================================================
# 4. Logprob and DPO Loss
# ============================================================

def move_nested_to_device(batch_part: Dict[str, torch.Tensor], device):
    return {
        k: v.to(device)
        for k, v in batch_part.items()
    }


def sequence_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    计算 response 部分的 sequence log probability。

    labels:
      prompt tokens = -100
      response tokens = token ids

    Causal LM:
      logits[:, t] 预测 input_ids[:, t+1]
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_mask = shift_labels != -100

    # 避免 gather 时出现 -100
    safe_labels = shift_labels.masked_fill(~loss_mask, 0)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_logps = token_logps * loss_mask

    seq_logps = token_logps.sum(dim=-1)

    if average_log_prob:
        lengths = loss_mask.sum(dim=-1).clamp(min=1)
        seq_logps = seq_logps / lengths

    return seq_logps


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float,
):
    """
    DPO loss:
    -log sigmoid(beta * [(pi_c - pi_r) - (ref_c - ref_r)])
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)

    losses = -F.logsigmoid(logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    reward_acc = (chosen_rewards > rejected_rewards).float()

    return losses, chosen_rewards, rejected_rewards, reward_acc, logits


# ============================================================
# 5. Reference Logprob Cache
# ============================================================

def load_ref_cache(cache_path: str) -> Dict[str, Dict[str, float]]:
    path = Path(cache_path)
    if not path.exists():
        return {}

    cache = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                cache[item["id"]] = {
                    "ref_chosen_logp": float(item["ref_chosen_logp"]),
                    "ref_rejected_logp": float(item["ref_rejected_logp"]),
                }
            except Exception:
                continue

    print(f"[RefCache] Loaded {len(cache)} cached ref logps from {cache_path}")
    return cache


def save_ref_cache(cache: Dict[str, Dict[str, float]], cache_path: str):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample_id, values in cache.items():
            record = {
                "id": sample_id,
                "ref_chosen_logp": values["ref_chosen_logp"],
                "ref_rejected_logp": values["ref_rejected_logp"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[RefCache] Saved {len(cache)} ref logps to {cache_path}")


@torch.no_grad()
def precompute_ref_logps(
    model,
    tokenizer,
    pairs: List[Dict[str, Any]],
    args,
    cache_path: Optional[str] = None,
):
    """
    以初始 SFT 模型作为 reference model，预计算 chosen/rejected logprobs。
    """
    cache = {}
    if cache_path and not args.overwrite_ref_cache:
        cache = load_ref_cache(cache_path)

    missing_pairs = [
        p for p in pairs
        if p["id"] not in cache
    ]

    print(f"[RefLogps] Total pairs   : {len(pairs)}")
    print(f"[RefLogps] Cached pairs  : {len(cache)}")
    print(f"[RefLogps] Missing pairs : {len(missing_pairs)}")

    if missing_pairs:
        collator = DPODataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            include_ref_logps=False,
        )

        loader = DataLoader(
            DPOPairDataset(missing_pairs),
            batch_size=args.ref_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

        model.eval()

        for batch in tqdm(loader, desc="Precomputing reference logps"):
            chosen = move_nested_to_device(batch["chosen"], model.device)
            rejected = move_nested_to_device(batch["rejected"], model.device)

            ref_chosen = sequence_logps(
                model=model,
                input_ids=chosen["input_ids"],
                attention_mask=chosen["attention_mask"],
                labels=chosen["labels"],
                average_log_prob=args.average_log_prob,
            )

            ref_rejected = sequence_logps(
                model=model,
                input_ids=rejected["input_ids"],
                attention_mask=rejected["attention_mask"],
                labels=rejected["labels"],
                average_log_prob=args.average_log_prob,
            )

            for sid, c_logp, r_logp in zip(
                batch["ids"],
                ref_chosen.detach().cpu().tolist(),
                ref_rejected.detach().cpu().tolist(),
            ):
                cache[sid] = {
                    "ref_chosen_logp": float(c_logp),
                    "ref_rejected_logp": float(r_logp),
                }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cache_path:
            save_ref_cache(cache, cache_path)

    # 写回 pairs
    for p in pairs:
        p["ref_chosen_logp"] = cache[p["id"]]["ref_chosen_logp"]
        p["ref_rejected_logp"] = cache[p["id"]]["ref_rejected_logp"]

    model.train()

    return pairs


# ============================================================
# 6. Evaluation
# ============================================================

@torch.no_grad()
def evaluate_dpo(model, dataloader, args):
    model.eval()

    total_loss = 0.0
    total_dpo_loss = 0.0
    total_sft_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    total_count = 0

    for batch in tqdm(dataloader, desc="Evaluating DPO", leave=False):
        chosen = move_nested_to_device(batch["chosen"], model.device)
        rejected = move_nested_to_device(batch["rejected"], model.device)

        ref_chosen = batch["ref_chosen_logps"].to(model.device)
        ref_rejected = batch["ref_rejected_logps"].to(model.device)

        policy_chosen = sequence_logps(
            model=model,
            input_ids=chosen["input_ids"],
            attention_mask=chosen["attention_mask"],
            labels=chosen["labels"],
            average_log_prob=args.average_log_prob,
        )

        policy_rejected = sequence_logps(
            model=model,
            input_ids=rejected["input_ids"],
            attention_mask=rejected["attention_mask"],
            labels=rejected["labels"],
            average_log_prob=args.average_log_prob,
        )

        losses, chosen_rewards, rejected_rewards, reward_acc, logits = dpo_loss(
            policy_chosen_logps=policy_chosen,
            policy_rejected_logps=policy_rejected,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            beta=args.beta,
        )

        dpo_loss_value = losses.mean()
        sft_loss_value = -policy_chosen.mean()
        combined_loss = dpo_loss_value + args.sft_loss_weight * sft_loss_value

        batch_size = len(batch["ids"])
        total_loss += combined_loss.item() * batch_size
        total_dpo_loss += dpo_loss_value.item() * batch_size
        total_sft_loss += sft_loss_value.item() * batch_size
        total_acc += reward_acc.mean().item() * batch_size
        total_margin += (chosen_rewards - rejected_rewards).mean().item() * batch_size
        total_count += batch_size

    model.train()

    if total_count == 0:
        return {
            "eval_loss": 0.0,
            "eval_dpo_loss": 0.0,
            "eval_sft_loss": 0.0,
            "eval_reward_acc": 0.0,
            "eval_reward_margin": 0.0,
        }

    return {
        "eval_loss": total_loss / total_count,
        "eval_dpo_loss": total_dpo_loss / total_count,
        "eval_sft_loss": total_sft_loss / total_count,
        "eval_reward_acc": total_acc / total_count,
        "eval_reward_margin": total_margin / total_count,
    }


# ============================================================
# 7. Training
# ============================================================

def save_adapter_and_tokenizer(model, tokenizer, output_dir: str, step_name: str):
    save_dir = Path(output_dir) / step_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"[Saving] Saved adapter to: {save_dir}")


def train_dpo(args):
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.base_model_path)
    model = load_policy_model(args)

    all_pairs = load_jsonl(args.dpo_file)

    train_pairs, eval_pairs = split_train_eval(
        data=all_pairs,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    # reference model = 初始 SFT 模型
    ref_cache_path = args.ref_cache_file
    if ref_cache_path is None:
        ref_cache_path = str(Path(args.output_dir) / "ref_logps_cache.jsonl")

    train_pairs = precompute_ref_logps(
        model=model,
        tokenizer=tokenizer,
        pairs=train_pairs,
        args=args,
        cache_path=ref_cache_path,
    )

    if eval_pairs:
        eval_pairs = precompute_ref_logps(
            model=model,
            tokenizer=tokenizer,
            pairs=eval_pairs,
            args=args,
            cache_path=ref_cache_path,
        )

    train_dataset = DPOPairDataset(train_pairs)
    eval_dataset = DPOPairDataset(eval_pairs) if eval_pairs else None

    train_collator = DPODataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        include_ref_logps=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=args.dataloader_num_workers,
    )

    eval_loader = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=train_collator,
            num_workers=args.dataloader_num_workers,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps,
    )

    print("\n" + "=" * 80)
    print("DPO Training Configuration")
    print("=" * 80)
    print(f"Train pairs                  : {len(train_dataset)}")
    print(f"Eval pairs                   : {len(eval_dataset) if eval_dataset else 0}")
    print(f"Epochs                       : {args.num_train_epochs}")
    print(f"Train batch size             : {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps  : {args.gradient_accumulation_steps}")
    print(f"Max train steps              : {max_train_steps}")
    print(f"Learning rate                : {args.learning_rate}")
    print(f"Beta                         : {args.beta}")
    print(f"Average log prob             : {args.average_log_prob}")
    print(f"SFT loss weight              : {args.sft_loss_weight}")
    print("=" * 80)

    global_step = 0
    best_eval_loss = float("inf")

    log_path = Path(args.output_dir) / "dpo_train_log.jsonl"

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{int(args.num_train_epochs)}")

        running_loss = 0.0
        running_dpo_loss = 0.0
        running_sft_loss = 0.0
        running_acc = 0.0
        running_margin = 0.0
        running_count = 0

        for step, batch in enumerate(epoch_iterator, start=1):
            chosen = move_nested_to_device(batch["chosen"], model.device)
            rejected = move_nested_to_device(batch["rejected"], model.device)

            ref_chosen = batch["ref_chosen_logps"].to(model.device)
            ref_rejected = batch["ref_rejected_logps"].to(model.device)

            policy_chosen = sequence_logps(
                model=model,
                input_ids=chosen["input_ids"],
                attention_mask=chosen["attention_mask"],
                labels=chosen["labels"],
                average_log_prob=args.average_log_prob,
            )

            policy_rejected = sequence_logps(
                model=model,
                input_ids=rejected["input_ids"],
                attention_mask=rejected["attention_mask"],
                labels=rejected["labels"],
                average_log_prob=args.average_log_prob,
            )

            losses, chosen_rewards, rejected_rewards, reward_acc, logits = dpo_loss(
                policy_chosen_logps=policy_chosen,
                policy_rejected_logps=policy_rejected,
                ref_chosen_logps=ref_chosen,
                ref_rejected_logps=ref_rejected,
                beta=args.beta,
            )

            dpo_loss_value = losses.mean()

            # DPO-v3: add a small supervised regularization term on the chosen response.
            # This helps keep the DPO-updated model close to the reference-style annotation
            # distribution learned during SFT. It is especially useful when DPO pairs are
            # reference-heavy and the goal is to preserve BLEU/ROUGE/CIDEr.
            # Recommended: use together with --average_log_prob.
            sft_loss_value = -policy_chosen.mean()
            loss = dpo_loss_value + args.sft_loss_weight * sft_loss_value

            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            batch_size = len(batch["ids"])
            running_loss += loss.item() * batch_size
            running_dpo_loss += dpo_loss_value.item() * batch_size
            running_sft_loss += sft_loss_value.item() * batch_size
            running_acc += reward_acc.mean().item() * batch_size
            running_margin += (chosen_rewards - rejected_rewards).mean().item() * batch_size
            running_count += batch_size

            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / max(1, running_count)
                    avg_dpo_loss = running_dpo_loss / max(1, running_count)
                    avg_sft_loss = running_sft_loss / max(1, running_count)
                    avg_acc = running_acc / max(1, running_count)
                    avg_margin = running_margin / max(1, running_count)

                    log_record = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "train_dpo_loss": avg_dpo_loss,
                        "train_sft_loss": avg_sft_loss,
                        "train_reward_acc": avg_acc,
                        "train_reward_margin": avg_margin,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }

                    print(
                        f"[Step {global_step}] "
                        f"loss={avg_loss:.4f}, "
                        f"dpo={avg_dpo_loss:.4f}, "
                        f"sft={avg_sft_loss:.4f}, "
                        f"reward_acc={avg_acc:.4f}, "
                        f"reward_margin={avg_margin:.4f}, "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

                    running_loss = 0.0
                    running_dpo_loss = 0.0
                    running_sft_loss = 0.0
                    running_acc = 0.0
                    running_margin = 0.0
                    running_count = 0

                if eval_loader is not None and global_step % args.eval_steps == 0:
                    eval_metrics = evaluate_dpo(model, eval_loader, args)

                    print(
                        f"[Eval step {global_step}] "
                        f"eval_loss={eval_metrics['eval_loss']:.4f}, "
                        f"eval_dpo={eval_metrics['eval_dpo_loss']:.4f}, "
                        f"eval_sft={eval_metrics['eval_sft_loss']:.4f}, "
                        f"eval_reward_acc={eval_metrics['eval_reward_acc']:.4f}, "
                        f"eval_reward_margin={eval_metrics['eval_reward_margin']:.4f}"
                    )

                    eval_record = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        **eval_metrics,
                    }

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        save_adapter_and_tokenizer(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=args.output_dir,
                            step_name="best"
                        )

                if global_step % args.save_steps == 0:
                    save_adapter_and_tokenizer(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=args.output_dir,
                        step_name=f"checkpoint-{global_step}"
                    )

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    print("\n[Saving] Saving final DPO adapter...")
    save_adapter_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        step_name="final"
    )

    # 如果没有 eval，就把 final 也视为 best
    if eval_loader is None:
        save_adapter_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            step_name="best"
        )

    summary = {
        "base_model_path": args.base_model_path,
        "sft_adapter_path": args.sft_adapter_path,
        "dpo_file": args.dpo_file,
        "output_dir": args.output_dir,
        "num_pairs_total": len(all_pairs),
        "num_train_pairs": len(train_dataset),
        "num_eval_pairs": len(eval_dataset) if eval_dataset else 0,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "sft_loss_weight": args.sft_loss_weight,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": max_train_steps,
        "average_log_prob": args.average_log_prob,
        "best_eval_loss": best_eval_loss if best_eval_loss < float("inf") else None,
    }

    summary_path = Path(args.output_dir) / "dpo_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Summary] Saved to {summary_path}")
    print("\nDPO training completed.")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/data/wb/models/Qwen2.5-Coder-7B-Instruct/",
    )
    parser.add_argument(
        "--sft_adapter_path",
        type=str,
        default="outputs/prefscom_sft_qwen2_5_coder_7b",
    )
    parser.add_argument(
        "--dpo_file",
        type=str,
        default="data/dpo/dpo_pairs.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/prefscom_dpo_qwen2_5_coder_7b",
    )
    parser.add_argument(
        "--ref_cache_file",
        type=str,
        default=None,
        help="Reference logprob cache path. Default: output_dir/ref_logps_cache.jsonl",
    )
    parser.add_argument("--overwrite_ref_cache", action="store_true")

    # model loading
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # sequence length
    parser.add_argument("--max_length", type=int, default=1152)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=128)

    # DPO training hyperparameters
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--ref_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument(
        "--average_log_prob",
        action="store_true",
        help="Use average response logprob instead of summed logprob. Default uses summed logprob, standard DPO style.",
    )
    parser.add_argument(
        "--sft_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of supervised regularization on chosen responses. "
            "For DPO-v3, use 0.05~0.10 together with --average_log_prob."
        ),
    )

    # eval / save / log
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.sft_loss_weight > 0 and not args.average_log_prob:
        print(
            "[Warning] --sft_loss_weight is enabled without --average_log_prob. "
            "This is valid, but summed logprob may make the SFT regularization length-biased. "
            "For DPO-v3, --average_log_prob is recommended."
        )

    print("=" * 80)
    print("PrefSCom: Train DPO Model")
    print("=" * 80)
    print(f"Base model       : {args.base_model_path}")
    print(f"SFT adapter      : {args.sft_adapter_path}")
    print(f"DPO file         : {args.dpo_file}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Load in 4bit     : {args.load_in_4bit}")
    print(f"bf16             : {args.bf16}")
    print(f"fp16             : {args.fp16}")
    print(f"beta             : {args.beta}")
    print(f"sft_loss_weight  : {args.sft_loss_weight}")
    print("=" * 80)

    train_dpo(args)


if __name__ == "__main__":
    main()