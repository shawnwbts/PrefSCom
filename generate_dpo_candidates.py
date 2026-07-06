
"""
PrefSCom: Generate DPO Candidate Comments

功能：
1. 读取 dpo_pool.jsonl
2. 对每个 Solidity 函数生成多个候选注释
3. 候选来源包括：
   - reference comment
   - base model zero-shot
   - SFT model greedy
   - SFT model sampling, temperature=0.3
   - SFT model sampling, temperature=0.6
   - SFT model sampling, temperature=0.9
   - rule-based hard negatives
4. 保存到 candidate_comments.jsonl

说明：
- 本脚本只负责生成候选，不直接构造 DPO chosen/rejected pair。
- 后续应对 candidate_comments.jsonl 中的候选计算 SIDE、semantic_score、
  consistency_score、security_sensitivity_score 等分数，再构造 DPO pairs。
"""

import gc
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    """
    兼容三种格式：

    格式 A：SFT-style
    {
      "instruction": "...",
      "input": "function ...",
      "output": "comment ..."
    }

    格式 B：old DPO-pool-style
    {
      "prompt": "...",
      "reference": "...",
      "token_seq": "function ..."
    }

    格式 C：new DPO-pool-style
    {
      "id": "dpo_pool_0",
      "instruction": "...",
      "function": "function ...",
      "reference": "...",
      "token_seq": "function ...",
      "source_file": "..."
    }
    """

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

            sample_id = item.get("id", f"dpo_pool_{line_no}")

            # 格式 A：SFT jsonl
            if all(k in item for k in ["instruction", "input", "output"]):
                instruction = str(item["instruction"]).strip()
                function_code = str(item["input"]).strip()
                reference = str(item["output"]).strip()

            # 格式 C：新版 dpo_pool.jsonl
            elif all(k in item for k in ["instruction", "function", "reference"]):
                instruction = str(item["instruction"]).strip()
                function_code = str(item["function"]).strip()
                reference = str(item["reference"]).strip()

                # 兜底：如果 function 字段为空，就用 token_seq
                if not function_code and "token_seq" in item:
                    function_code = str(item["token_seq"]).strip()

            # 格式 B：旧版 dpo_pool.jsonl
            elif all(k in item for k in ["prompt", "reference", "token_seq"]):
                instruction = (
                    "Generate a concise, faithful, and security-aware comment "
                    "for the given Solidity function."
                )
                function_code = str(item["token_seq"]).strip()
                reference = str(item["reference"]).strip()

            else:
                print(f"[Warning] Missing required fields at line {line_no}: {item.keys()}")
                continue

            if not function_code or not reference:
                continue

            sample = {
                "id": sample_id,
                "instruction": instruction,
                "function": function_code,
                "reference": reference,
            }

            # 保留辅助字段，方便后续溯源
            if "source_file" in item:
                sample["source_file"] = str(item["source_file"]).strip()
            if "token_seq" in item:
                sample["token_seq"] = str(item["token_seq"]).strip()

            samples.append(sample)

    print(f"[Data] Loaded {len(samples)} samples from {data_path}")
    return samples

# ============================================================
# 2. Prompt Construction
# ============================================================

def build_prompt(tokenizer, instruction: str, function_code: str) -> str:
    """
    与 SFT 训练阶段保持一致。
    """

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

def load_tokenizer(base_model_path: str):
    print(f"[Tokenizer] Loading tokenizer from: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return tokenizer


def load_model(
    base_model_path: str,
    adapter_path: Optional[str],
    load_in_4bit: bool = True,
    bf16: bool = True,
):
    if load_in_4bit:
        print("[Model] Loading model in 4-bit mode.")
        compute_dtype = torch.bfloat16 if bf16 else torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    print(f"[Model] Loading base model from: {base_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )

    if adapter_path:
        print(f"[PEFT] Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

    model.eval()
    return model


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 4. Output Cleaning
# ============================================================

def clean_prediction(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    bad_prefixes = [
        "assistant",
        "Assistant:",
        "Comment:",
        "### Comment:",
        "Here is the comment:",
        "The comment is:",
        "Generated comment:",
        "Output:",
        "Answer:",
    ]

    for prefix in bad_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = text.replace("```solidity", "").replace("```", "").strip()

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        text = lines[0]

    if len(text) >= 2 and text[0] in ['"', "'"] and text[-1] == text[0]:
        text = text[1:-1].strip()

    # 去掉 Solidity/NatSpec 残留符号
    text = text.replace("/**", "").replace("/*", "").replace("*/", "")
    text = text.replace("///", "").replace("//", "")
    text = re.sub(r"^\s*\*\s*", "", text)

    # 如果模型输出了 @dev，去掉标签，仅保留内容
    text = re.sub(r"^\s*@dev\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*@notice\s+", "", text, flags=re.IGNORECASE)

    # 如果输出中包含 @param/@return，说明模型多生成了，截断到这些标签之前
    text = re.split(r"\s+@(param|return|returns)\b", text, flags=re.IGNORECASE)[0].strip()

    # 避免过长输出，截断到第一句
    for sep in [". ", "? ", "! "]:
        if sep in text:
            text = text.split(sep)[0].strip() + sep.strip()
            break

    text = " ".join(text.split()).strip()

    return text


def normalize_for_dedup(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def add_candidate(
    candidates: List[Dict[str, Any]],
    source: str,
    comment: str,
    role_hint: str = "model_candidate",
    extra: Optional[Dict[str, Any]] = None,
):
    comment = clean_prediction(comment)

    if not comment:
        return

    existing = {normalize_for_dedup(c["comment"]) for c in candidates}
    key = normalize_for_dedup(comment)

    if key in existing:
        return

    item = {
        "source": source,
        "comment": comment,
        "role_hint": role_hint,
    }

    if extra:
        item.update(extra)

    candidates.append(item)


# ============================================================
# 5. Rule-based Hard Negative Generation
# ============================================================

def detect_security_facts(function_code: str) -> Dict[str, bool]:
    code_raw = str(function_code)
    code = code_raw.lower()

    facts = {
        "access_control": any(x in code for x in [
            "onlyowner", "only owner", "hasrole", "msg.sender == owner",
            "msg.sender==owner", "owner()", "_owner", "admin", "authorized",
            "onlyadmin", "onlygovernance", "onlyoperator"
        ]),
        "time_constraint": any(x in code for x in [
            "block.timestamp", "block.number", "deadline", "locktime",
            "unlocktime", "timelock", "starttime", "endtime", "duration"
        ]),
        "fund_transfer": any(x in code for x in [
            ".transfer(", ".send(", "call{value", "call.value",
            "ierc20", "safeTransfer".lower(), "safetransfer", "transferfrom",
            "transferFrom".lower(), "approve", "allowance", "balance"
        ]),
        "external_call": any(x in code for x in [
            ".call(", ".delegatecall(", ".staticcall(", "delegatecall",
            "interface", "callback", "onerc", "onerc721", "onerc1155"
        ]),
        "state_validation": any(x in code for x in [
            "require(", "require (", "revert(", "revert (", "assert(",
            "assert (", "modifier"
        ]),
        "emergency_control": any(x in code for x in [
            "paused", "pause", "unpause", "whenpaused", "whennotpaused",
            "blacklist", "whitelist", "emergency"
        ]),
        "unchecked_arithmetic": "unchecked" in code,
        "mint": "mint" in code,
        "burn": "burn" in code,
        "approve_or_allowance": any(x in code for x in ["approve", "allowance", "spender"]),
        "delegatecall": "delegatecall" in code,
    }

    return facts


def remove_security_terms(text: str, terms: List[str]) -> str:
    """
    从 reference 中移除部分安全/权限词，构造 omission negative。
    """

    text = clean_prediction(text)

    for term in terms:
        text = re.sub(rf"\b{re.escape(term)}\b", "", text, flags=re.IGNORECASE)

    text = " ".join(text.split()).strip()

    if not text or len(text.split()) < 3:
        return "Performs the requested operation."

    return text


def generate_hard_negatives(
    function_code: str,
    reference: str,
    max_negatives: int = 4
) -> List[Dict[str, str]]:
    """
    生成面向 DPO 的 hard negatives。
    目标不是生成随机差注释，而是生成“看似合理但不忠实或缺少安全语义”的注释。
    """

    facts = detect_security_facts(function_code)
    code = str(function_code).lower()
    ref = clean_prediction(reference)
    ref_lower = ref.lower()

    negatives = []

    # 1. Generic incomplete negative
    negatives.append({
        "source": "hard_negative_generic_incomplete",
        "comment": "Executes the function.",
        "perturbation_type": "generic_incomplete",
    })

    # 2. Access-control reversal / omission
    if facts["access_control"]:
        if "owner" in ref_lower:
            neg = ref
            neg = re.sub(r"\bonly the owner\b", "any user", neg, flags=re.IGNORECASE)
            neg = re.sub(r"\bowner\b", "user", neg, flags=re.IGNORECASE)
        elif "admin" in ref_lower:
            neg = re.sub(r"\badmin\b", "user", ref, flags=re.IGNORECASE)
        else:
            neg = "Allows any user to perform this operation."

        negatives.append({
            "source": "hard_negative_access_control",
            "comment": clean_prediction(neg),
            "perturbation_type": "access_control_reversal",
        })

        negatives.append({
            "source": "hard_negative_access_omission",
            "comment": remove_security_terms(
                ref,
                terms=["owner", "admin", "authorized", "only", "governance", "operator"]
            ),
            "perturbation_type": "access_control_omission",
        })

    # 3. Time constraint omission
    if facts["time_constraint"]:
        negatives.append({
            "source": "hard_negative_time_constraint",
            "comment": "Performs the operation at any time.",
            "perturbation_type": "time_constraint_omission",
        })

    # 4. Asset-flow omission
    if facts["fund_transfer"]:
        negatives.append({
            "source": "hard_negative_asset_flow_omission",
            "comment": "Updates internal records without transferring assets.",
            "perturbation_type": "asset_flow_omission",
        })

    # 5. External-call omission
    if facts["external_call"]:
        negatives.append({
            "source": "hard_negative_external_call_omission",
            "comment": "Performs only internal state updates without external contract interaction.",
            "perturbation_type": "external_call_omission",
        })

    # 6. State validation omission
    if facts["state_validation"]:
        negatives.append({
            "source": "hard_negative_validation_omission",
            "comment": "Executes the operation without checking any preconditions.",
            "perturbation_type": "validation_omission",
        })

    # 7. Pause / emergency omission
    if facts["emergency_control"]:
        negatives.append({
            "source": "hard_negative_emergency_omission",
            "comment": "Executes the operation without considering paused or emergency states.",
            "perturbation_type": "emergency_control_omission",
        })

    # 8. Mint / burn action reversal
    if facts["mint"]:
        negatives.append({
            "source": "hard_negative_action_reverse",
            "comment": "Burns tokens from the caller.",
            "perturbation_type": "mint_burn_reversal",
        })

    if facts["burn"]:
        negatives.append({
            "source": "hard_negative_action_reverse",
            "comment": "Mints new tokens to the caller.",
            "perturbation_type": "mint_burn_reversal",
        })

    # 9. Approval misread
    if facts["approve_or_allowance"]:
        negatives.append({
            "source": "hard_negative_approval_misread",
            "comment": "Transfers tokens directly to the spender.",
            "perturbation_type": "approval_as_transfer",
        })

    # 10. Delegatecall semantic error
    if facts["delegatecall"]:
        negatives.append({
            "source": "hard_negative_delegatecall_misread",
            "comment": "Calls the implementation without preserving the caller context.",
            "perturbation_type": "delegatecall_semantic_error",
        })

    # 11. Unchecked arithmetic hallucination
    if facts["unchecked_arithmetic"]:
        negatives.append({
            "source": "hard_negative_unchecked_arithmetic",
            "comment": "Performs arithmetic operations with complete overflow protection.",
            "perturbation_type": "unchecked_arithmetic_hallucination",
        })

    # 去重、清洗、截断
    unique = []
    seen = set()

    for item in negatives:
        comment = clean_prediction(item["comment"])
        if not comment:
            continue

        key = normalize_for_dedup(comment)
        if key in seen:
            continue

        item["comment"] = comment
        unique.append(item)
        seen.add(key)

    return unique[:max_negatives]


# ============================================================
# 6. Batch Generation
# ============================================================

def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    num_return_sequences: int,
    desc: str,
) -> List[List[str]]:
    """
    返回格式：
    [
      [output_1_for_sample_1, output_2_for_sample_1],
      [output_1_for_sample_2, output_2_for_sample_2],
      ...
    ]
    """

    all_results: List[List[str]] = []

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "num_return_sequences": num_return_sequences,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": repetition_penalty,
    }

    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start:start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )

        input_width = inputs["input_ids"].shape[1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )

        batch_results = [[] for _ in range(len(batch_prompts))]

        for i, output_ids in enumerate(outputs):
            sample_idx = i // num_return_sequences

            generated_ids = output_ids[input_width:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            text = clean_prediction(text)

            if sample_idx < len(batch_results):
                batch_results[sample_idx].append(text)

        all_results.extend(batch_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# ============================================================
# 7. Candidate Generation Pipeline
# ============================================================

def load_done_ids(output_file: str) -> set:
    done = set()
    path = Path(output_file)

    if not path.exists():
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["id"])
            except Exception:
                continue

    return done


def generate_candidates(args):
    set_seed(args.seed)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(args.dpo_pool_file, encoding=args.data_encoding)

    if args.resume and output_path.exists():
        done_ids = load_done_ids(args.output_file)
        print(f"[Resume] Found {len(done_ids)} completed samples.")
        samples = [s for s in samples if s["id"] not in done_ids]

    if args.max_samples is not None:
        samples = samples[:args.max_samples]
        print(f"[Debug] max_samples={args.max_samples}, remaining samples={len(samples)}")

    if not samples:
        print("[Info] No samples to process.")
        return

    tokenizer = load_tokenizer(args.base_model_path)

    prompts = [
        build_prompt(tokenizer, s["instruction"], s["function"])
        for s in samples
    ]

    candidate_records: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        candidates = []

        # reference candidate
        if args.include_reference:
            add_candidate(
                candidates,
                source="reference",
                comment=sample["reference"],
                role_hint="positive_anchor",
            )

        # rule-based hard negatives
        if args.include_hard_negatives:
            hard_negs = generate_hard_negatives(
                function_code=sample["function"],
                reference=sample["reference"],
                max_negatives=args.num_hard_negatives,
            )

            for neg in hard_negs:
                add_candidate(
                    candidates,
                    source=neg["source"],
                    comment=neg["comment"],
                    role_hint="negative_only",
                    extra={"perturbation_type": neg["perturbation_type"]}
                )

        candidate_records[sample["id"]] = {
            "id": sample["id"],
            "instruction": sample["instruction"],
            "function": sample["function"],
            "reference": sample["reference"],
            "candidates": candidates,
        }

    # ------------------------------------------------------------
    # A. Base model zero-shot
    # ------------------------------------------------------------

    if args.include_base_zero_shot:
        print("\n" + "=" * 80)
        print("[Stage A] Generating base model zero-shot candidates")
        print("=" * 80)

        base_model = load_model(
            base_model_path=args.base_model_path,
            adapter_path=None,
            load_in_4bit=args.load_in_4bit,
            bf16=args.bf16,
        )

        base_outputs = batch_generate(
            model=base_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            do_sample=False,
            temperature=0.2,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=1,
            desc="Base zero-shot",
        )

        for sample, outs in zip(samples, base_outputs):
            if outs:
                add_candidate(
                    candidate_records[sample["id"]]["candidates"],
                    source="base_zero_shot",
                    comment=outs[0],
                    role_hint="model_candidate",
                    extra={
                        "decode_strategy": "greedy",
                        "model_stage": "base"
                    }
                )

        release_model(base_model)

    # ------------------------------------------------------------
    # B. SFT model candidates
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("[Stage B] Generating SFT model candidates")
    print("=" * 80)

    sft_model = load_model(
        base_model_path=args.base_model_path,
        adapter_path=args.sft_adapter_path,
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
    )

    # B1. SFT greedy
    if args.include_sft_greedy:
        greedy_outputs = batch_generate(
            model=sft_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            do_sample=False,
            temperature=0.2,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=1,
            desc="SFT greedy",
        )

        for sample, outs in zip(samples, greedy_outputs):
            if outs:
                add_candidate(
                    candidate_records[sample["id"]]["candidates"],
                    source="sft_greedy",
                    comment=outs[0],
                    role_hint="model_candidate",
                    extra={
                        "decode_strategy": "greedy",
                        "model_stage": "sft"
                    }
                )

    # B2. SFT sampling with multiple temperatures
    sampling_configs = [
        ("sft_temp_0.3", 0.3, args.num_sft_temp03),
        ("sft_temp_0.6", 0.6, args.num_sft_temp06),
        ("sft_temp_0.9", 0.9, args.num_sft_temp09),
    ]

    for source_prefix, temperature, num_return_sequences in sampling_configs:
        if num_return_sequences <= 0:
            continue

        sampled_outputs = batch_generate(
            model=sft_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            do_sample=True,
            temperature=temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=num_return_sequences,
            desc=f"SFT sampling temp={temperature}",
        )

        for sample, outs in zip(samples, sampled_outputs):
            for i, out in enumerate(outs, start=1):
                add_candidate(
                    candidate_records[sample["id"]]["candidates"],
                    source=f"{source_prefix}_{i}",
                    comment=out,
                    role_hint="model_candidate",
                    extra={
                        "temperature": temperature,
                        "decode_strategy": "sampling",
                        "model_stage": "sft"
                    }
                )

    release_model(sft_model)

    # ------------------------------------------------------------
    # C. Save final records
    # ------------------------------------------------------------

    mode = "a" if args.resume else "w"

    with open(output_path, mode, encoding="utf-8") as fout:
        for sample in samples:
            record = candidate_records[sample["id"]]
            record["num_candidates"] = len(record["candidates"])

            source_counts = {}
            role_counts = {}

            for c in record["candidates"]:
                source_counts[c["source"]] = source_counts.get(c["source"], 0) + 1
                role_counts[c["role_hint"]] = role_counts.get(c["role_hint"], 0) + 1

            record["source_counts"] = source_counts
            record["role_counts"] = role_counts

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[Done] Candidate comments saved to: {output_path}")
    print(f"[Done] Processed samples: {len(samples)}")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Paths
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
        "--dpo_pool_file",
        type=str,
        default="data/sft/dpo_pool.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/dpo/candidate_comments.jsonl",
    )
    parser.add_argument("--data_encoding", type=str, default="utf-8")

    # Candidate source switches
    parser.add_argument("--include_reference", action="store_true", default=True)
    parser.add_argument("--include_hard_negatives", action="store_true", default=True)
    parser.add_argument("--include_base_zero_shot", action="store_true", default=True)
    parser.add_argument("--include_sft_greedy", action="store_true", default=True)

    # Number of sampled candidates
    parser.add_argument("--num_sft_temp03", type=int, default=1)
    parser.add_argument("--num_sft_temp06", type=int, default=2)
    parser.add_argument("--num_sft_temp09", type=int, default=2)
    parser.add_argument("--num_hard_negatives", type=int, default=4)

    # Generation config
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # Model loading config
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Running config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    # 如果用户显式指定 fp16，则关闭 bf16
    if args.fp16:
        args.bf16 = False

    print("=" * 80)
    print("PrefSCom: Generate DPO Candidate Comments")
    print("=" * 80)
    print(f"Base model      : {args.base_model_path}")
    print(f"SFT adapter     : {args.sft_adapter_path}")
    print(f"DPO pool file   : {args.dpo_pool_file}")
    print(f"Output file     : {args.output_file}")
    print(f"Load in 4bit    : {args.load_in_4bit}")
    print(f"bf16            : {args.bf16}")
    print(f"temp=0.3        : {args.num_sft_temp03}")
    print(f"temp=0.6        : {args.num_sft_temp06}")
    print(f"temp=0.9        : {args.num_sft_temp09}")
    print(f"hard negatives  : {args.num_hard_negatives}")
    print("=" * 80)

    generate_candidates(args)


if __name__ == "__main__":
    main()