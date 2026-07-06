
"""
PrefSCom: Candidate Scoring

功能：
1. 读取 data/dpo/candidate_comments.jsonl
2. 对每个 candidate comment 计算：
   - F_code: SIDE(function, candidate_comment)
   - F_sem: SentenceBERT(candidate_comment, reference_comment)
   - F_sec: SecurityCoverage(function, candidate_comment)
   - F_style: StyleScore(candidate_comment)
   - H: HallucinationPenalty(function, candidate_comment)
3. 计算综合偏好分数 Score
4. 保存 data/dpo/scored_candidates.jsonl


"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Basic Utils
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error at line {line_no}: {e}")

    print(f"[Data] Loaded {len(data)} records from {path}")
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Data] Saved {len(data)} records to {path}")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def clamp_01(x: float) -> float:
    """
    SIDE / SentenceBERT cosine similarity 通常在 [-1, 1]。
    这里为了和其他 0-1 指标组合，直接裁剪到 [0, 1]。
    不使用 (x+1)/2，避免把负相关样本抬高。
    """
    return max(0.0, min(1.0, float(x)))


# ============================================================
# 2. SIDE Model Similarity
# ============================================================

def mean_pooling(model_output, attention_mask):
    """
    和你给出的 SIDE 计算方式保持一致。
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )


def encode_with_hf_model(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 256,
    desc: str = "Encoding"
) -> torch.Tensor:
    """
    使用 AutoModel + mean pooling 编码文本。
    用于 SIDE 模型。
    """
    all_embeddings = []

    model.eval()

    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[start:start + batch_size]

        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output,
            encoded_input["attention_mask"]
        )

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        all_embeddings.append(sentence_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_side_scores(
    pairs: List[Tuple[str, str]],
    side_model_path: str,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 256,
) -> List[float]:
    """
    计算 SIDE(function, candidate_comment)。

    pairs:
        [(function_code, candidate_comment), ...]
    """
    print(f"\n[SIDE] Loading SIDE model from: {side_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(side_model_path)
    model = AutoModel.from_pretrained(side_model_path).to(device)
    model.eval()

    left_texts = [p[0] for p in pairs]
    right_texts = [p[1] for p in pairs]

    left_emb = encode_with_hf_model(
        texts=left_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        desc="[SIDE] Encoding functions"
    )

    right_emb = encode_with_hf_model(
        texts=right_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        desc="[SIDE] Encoding comments"
    )

    sims = torch.sum(left_emb * right_emb, dim=1).tolist()

    # 释放内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [float(x) for x in sims]


# ============================================================
# 3. SentenceBERT Semantic Similarity
# ============================================================

def compute_sentence_semantic_scores(
    pairs: List[Tuple[str, str]],
    sem_model_path: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> List[float]:
    """
    计算 F_sem = SentenceBERT(candidate_comment, reference_comment)。

    pairs:
        [(candidate_comment, reference_comment), ...]
    """
    print(f"\n[SEM] Loading SentenceTransformer from: {sem_model_path}")

    model = SentenceTransformer(sem_model_path, device=device)

    left_texts = [p[0] for p in pairs]
    right_texts = [p[1] for p in pairs]

    print("[SEM] Encoding candidate comments...")
    left_emb = model.encode(
        left_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("[SEM] Encoding reference comments...")
    right_emb = model.encode(
        right_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    sims = torch.sum(left_emb * right_emb, dim=1).detach().cpu().tolist()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [float(x) for x in sims]


# ============================================================
# 4. Security Facts Extraction
# ============================================================

def extract_security_facts(function_code: str) -> List[Dict[str, Any]]:
    """
    从 Solidity 函数中用规则抽取 security-relevant facts。

    注意：
    这是启发式抽取，不等价于漏洞检测。
    只用于衡量注释是否覆盖安全相关语义。
    """
    code = function_code or ""
    code_lower = code.lower()

    facts = []

    # 1. Access Control
    if any(x in code_lower for x in [
        "onlyowner",
        "hasrole",
        "msg.sender == owner",
        "msg.sender==owner",
        "owner()",
        "_owner",
        "admin",
        "authorized",
        "auth",
    ]):
        facts.append({
            "type": "access_control",
            "description": "access control or caller permission",
            "cover_terms": [
                "owner", "admin", "role", "permission", "authorized",
                "only", "caller", "sender", "access"
            ]
        })

    # 2. State Validation / Preconditions
    if any(x in code_lower for x in [
        "require(",
        "revert(",
        "assert(",
        "modifier",
    ]):
        facts.append({
            "type": "state_validation",
            "description": "precondition or state validation",
            "cover_terms": [
                "require", "revert", "assert", "check", "ensure",
                "validate", "must", "condition", "precondition",
                "only if", "when", "unless", "valid"
            ]
        })

    # 3. Time Constraint
    if any(x in code_lower for x in [
        "block.timestamp",
        "block.number",
        "deadline",
        "locktime",
        "unlocktime",
        "time lock",
        "timelock",
        "starttime",
        "endtime",
        "duration",
        "period",
    ]):
        facts.append({
            "type": "time_constraint",
            "description": "time or block constraint",
            "cover_terms": [
                "time", "deadline", "lock", "unlock", "timestamp",
                "block", "before", "after", "expired", "duration",
                "period", "start", "end"
            ]
        })

    # 4. ETH / Token Transfer
    if any(x in code_lower for x in [
        ".transfer(",
        ".send(",
        "call{value",
        "call.value",
        "safetransfer",
        "transferfrom",
        "transferfrom(",
        "transfer(",
        "approve(",
        "allowance",
        "ierc20",
        "erc20",
        "erc721",
    ]):
        facts.append({
            "type": "fund_transfer",
            "description": "fund or token transfer/approval",
            "cover_terms": [
                "transfer", "send", "fund", "funds", "eth", "ether",
                "token", "tokens", "approve", "approval", "allowance",
                "spender", "recipient", "receiver", "payment"
            ]
        })

    # 5. External Interaction
    if any(x in code_lower for x in [
        ".call(",
        ".delegatecall(",
        ".staticcall(",
        "delegatecall",
        "callback",
        "onerc",
        "interface",
        "oracle",
    ]):
        facts.append({
            "type": "external_interaction",
            "description": "external contract interaction",
            "cover_terms": [
                "external", "call", "contract", "delegatecall",
                "staticcall", "callback", "receiver", "oracle",
                "interact", "interaction"
            ]
        })

    # 6. Authorization Logic
    if any(x in code_lower for x in [
        "approve(",
        "allowance",
        "transferfrom",
        "permit",
    ]):
        facts.append({
            "type": "authorization_logic",
            "description": "token approval or allowance authorization",
            "cover_terms": [
                "approve", "approval", "allowance", "spender",
                "authorized", "permission", "permit", "transferfrom"
            ]
        })

    # 7. Emergency Control
    if any(x in code_lower for x in [
        "paused",
        "pause",
        "unpause",
        "blacklist",
        "whitelist",
        "emergency",
    ]):
        facts.append({
            "type": "emergency_control",
            "description": "pause, blacklist, whitelist, or emergency control",
            "cover_terms": [
                "pause", "paused", "unpause", "blacklist", "whitelist",
                "emergency", "disabled", "enabled", "blocked"
            ]
        })

    # 8. Unchecked Arithmetic
    if "unchecked" in code_lower:
        facts.append({
            "type": "unchecked_arithmetic",
            "description": "unchecked arithmetic operation",
            "cover_terms": [
                "unchecked", "arithmetic", "overflow", "underflow"
            ]
        })

    return facts


def compute_security_coverage(
    function_code: str,
    comment: str
) -> Tuple[float, List[str], List[str], int]:
    """
    计算 F_sec = covered_security_facts / all_security_facts。

    返回：
    - f_sec
    - all_fact_types
    - covered_fact_types
    - num_facts
    """
    facts = extract_security_facts(function_code)

    if not facts:
        # 没有安全相关事实时，不惩罚该候选
        return 1.0, [], [], 0

    comment_lower = (comment or "").lower()

    covered = []

    for fact in facts:
        terms = fact["cover_terms"]
        if any(term.lower() in comment_lower for term in terms):
            covered.append(fact["type"])

    all_fact_types = [fact["type"] for fact in facts]

    f_sec = len(covered) / len(facts)

    return f_sec, all_fact_types, covered, len(facts)


# ============================================================
# 5. Hallucination Penalty
# ============================================================

def compute_hallucination_penalty(
    function_code: str,
    comment: str
) -> Tuple[float, List[str]]:
    """
    用规则检测明显的 unsupported / contradictory security claims。

    H 越大越差，范围 [0, 1]。
    """
    code = (function_code or "").lower()
    text = (comment or "").lower()

    flags = []

    has_access_control = any(x in code for x in [
        "onlyowner", "hasrole", "msg.sender == owner", "msg.sender==owner",
        "owner()", "_owner", "admin", "authorized"
    ])

    has_time_constraint = any(x in code for x in [
        "block.timestamp", "block.number", "deadline", "locktime",
        "unlocktime", "timelock", "starttime", "endtime", "duration"
    ])

    has_state_validation = any(x in code for x in [
        "require(", "revert(", "assert(", "modifier"
    ])

    has_fund_transfer = any(x in code for x in [
        ".transfer(", ".send(", "call{value", "call.value",
        "safetransfer", "transferfrom", "approve(", "allowance"
    ])

    has_external_interaction = any(x in code for x in [
        ".call(", ".delegatecall(", ".staticcall(", "delegatecall",
        "callback", "oracle"
    ])

    has_reentrancy_guard = any(x in code for x in [
        "nonreentrant", "_status", "reentrancyguard"
    ])

    has_unchecked = "unchecked" in code

    # 1. 声称 only owner / admin，但代码没有明显访问控制
    if not has_access_control:
        if any(x in text for x in [
            "only the owner", "owner only", "only owner",
            "admin only", "only admin", "restricted to owner",
            "restricted to the owner"
        ]):
            flags.append("unsupported_access_control_claim")

    # 2. 代码有时间约束，但注释说 anytime / no time restriction
    if has_time_constraint:
        if any(x in text for x in [
            "at any time", "anytime", "without time restriction",
            "no time restriction", "whenever"
        ]):
            flags.append("contradict_time_constraint")

    # 3. 代码有 require/revert，但注释说 no precondition
    if has_state_validation:
        if any(x in text for x in [
            "without checking", "without any check", "without precondition",
            "no precondition", "without validation"
        ]):
            flags.append("contradict_state_validation")

    # 4. 代码存在资金转移，但注释说不转移资金
    if has_fund_transfer:
        if any(x in text for x in [
            "without transferring funds",
            "without transferring tokens",
            "does not transfer",
            "no fund transfer"
        ]):
            flags.append("contradict_fund_transfer")

    # 5. 代码存在外部交互，但注释说没有外部交互
    if has_external_interaction:
        if any(x in text for x in [
            "without external interaction",
            "without external call",
            "only internal",
            "internal state updates only"
        ]):
            flags.append("contradict_external_interaction")

    # 6. 注释声称防重入，但代码没有明显 nonReentrant / guard
    if not has_reentrancy_guard:
        if any(x in text for x in [
            "prevents reentrancy",
            "reentrancy safe",
            "reentrancy-safe",
            "protects against reentrancy"
        ]):
            flags.append("unsupported_reentrancy_guarantee")

    # 7. 代码有 unchecked，但注释声称 complete overflow protection
    if has_unchecked:
        if any(x in text for x in [
            "complete overflow protection",
            "fully prevents overflow",
            "prevents overflow",
            "overflow safe"
        ]):
            flags.append("contradict_unchecked_arithmetic")

    # 8. 函数无 bool returns，但注释声称 returns true
    has_bool_return = bool(re.search(r"returns\s*\([^)]*bool", code)) or "returns (bool" in code
    if not has_bool_return:
        if any(x in text for x in [
            "returns true",
            "return true"
        ]):
            flags.append("unsupported_bool_return")

    owner_claim_patterns = [
        "owner can",
        "only owner",
        "only the owner",
        "owner may",
        "owner is able",
        "called by owner",
        "called by the owner",
        "can only be called by owner",
        "can only be called by the owner",
        "restricted to owner",
        "restricted to the owner",
    ]

    if not has_access_control:
        if any(p in text for p in owner_claim_patterns):
            flags.append("unsupported_owner_permission_claim")

    has_emergency_control = any(x in code for x in [
        "pause", "paused", "unpause", "emergency", "blacklist", "whitelist"
    ])

    if not has_emergency_control:
        if any(x in text for x in [
            "case of emergency",
            "in emergency",
            "emergency situation",
            "emergency use"
        ]):
            flags.append("unsupported_emergency_claim")

    # 每个明显幻觉给 0.25 惩罚，上限 1
    penalty = min(1.0, 0.25 * len(flags))

    return penalty, flags


# ============================================================
# 6. Style Score
# ============================================================

def compute_style_score(comment: str) -> Tuple[float, List[str]]:
    """
    简单评估注释是否简洁、自然、符合单句注释风格。
    """
    text = normalize_text(comment)
    lower = text.lower()
    flags = []

    if not text:
        return 0.0, ["empty"]

    words = text.split()
    word_count = len(words)

    score = 1.0

    # 长度过短或过长
    if word_count < 5:
        score -= 0.35
        flags.append("too_short")
    elif word_count < 8:
        score -= 0.15
        flags.append("slightly_short")

    if word_count > 40:
        score -= 0.35
        flags.append("too_long")
    elif word_count > 35:
        score -= 0.15
        flags.append("slightly_long")

    # 多句输出
    sentence_end_count = sum(text.count(x) for x in [".", "!", "?"])
    if sentence_end_count > 2:
        score -= 0.20
        flags.append("multi_sentence")

    # 包含代码/markdown
    if "function " in lower or "{" in text or "}" in text or "```" in text:
        score -= 0.30
        flags.append("contains_code_or_markdown")

    # 模板化废话
    bad_phrases = [
        "here is",
        "this comment",
        "the comment is",
        "generated comment",
        "as an ai",
    ]
    if any(p in lower for p in bad_phrases):
        score -= 0.25
        flags.append("template_phrase")

    # 没有句号不是严重问题，但略微扣分
    if not text.endswith((".", "!", "?")):
        score -= 0.05
        flags.append("no_sentence_end")

    return max(0.0, min(1.0, score)), flags


# ============================================================
# 7. Main Scoring Pipeline
# ============================================================

def flatten_candidates(records: List[Dict[str, Any]]):
    """
    将所有 candidate 展开，便于批量计算 SIDE 和 SentenceBERT。
    """
    flat = []

    for rec_idx, record in enumerate(records):
        function_code = record.get("function", "")
        reference = record.get("reference", "")

        candidates = record.get("candidates", [])

        for cand_idx, cand in enumerate(candidates):
            comment = cand.get("comment", "")

            if not normalize_text(comment):
                continue

            flat.append({
                "rec_idx": rec_idx,
                "cand_idx": cand_idx,
                "function": function_code,
                "reference": reference,
                "comment": comment,
            })

    return flat


def score_candidates(args):
    records = load_jsonl(args.candidate_file)

    flat = flatten_candidates(records)

    if not flat:
        raise ValueError("No valid candidates found.")

    print(f"[Scoring] Total candidates: {len(flat)}")

    # ------------------------------------------------------------
    # 1. Compute SIDE for F_code
    # ------------------------------------------------------------

    side_pairs = [
        (item["function"], item["comment"])
        for item in flat
    ]

    side_raw_scores = compute_side_scores(
        pairs=side_pairs,
        side_model_path=args.side_model_path,
        device=args.side_device,
        batch_size=args.side_batch_size,
        max_length=args.side_max_length,
    )

    # ------------------------------------------------------------
    # 2. Compute SentenceBERT similarity for F_sem
    # ------------------------------------------------------------

    sem_pairs = [
        (item["comment"], item["reference"])
        for item in flat
    ]

    sem_raw_scores = compute_sentence_semantic_scores(
        pairs=sem_pairs,
        sem_model_path=args.sem_model_path,
        device=args.sem_device,
        batch_size=args.sem_batch_size,
    )

    # ------------------------------------------------------------
    # 3. Rule-based scores and final score
    # ------------------------------------------------------------

    print("\n[Scoring] Computing rule-based scores and final preference scores...")

    for item, side_raw, sem_raw in tqdm(
        zip(flat, side_raw_scores, sem_raw_scores),
        total=len(flat),
        desc="Scoring candidates"
    ):
        rec_idx = item["rec_idx"]
        cand_idx = item["cand_idx"]

        function_code = item["function"]
        comment = item["comment"]

        f_code = clamp_01(side_raw)
        f_sem = clamp_01(sem_raw)

        f_sec, all_sec_facts, covered_sec_facts, num_sec_facts = compute_security_coverage(
            function_code=function_code,
            comment=comment
        )

        hallucination_penalty, hallucination_flags = compute_hallucination_penalty(
            function_code=function_code,
            comment=comment
        )

        f_style, style_flags = compute_style_score(comment)

        final_score = (
            args.w_code * f_code
            + args.w_sem * f_sem
            + args.w_sec * f_sec
            + args.w_style * f_style
            - args.w_hall * hallucination_penalty
        )

        final_score = max(0.0, min(1.0, final_score))

        cand = records[rec_idx]["candidates"][cand_idx]

        cand["side_raw"] = float(side_raw)
        cand["f_code"] = float(f_code)

        cand["sentence_sim_raw"] = float(sem_raw)
        cand["f_sem"] = float(f_sem)

        cand["security_facts"] = all_sec_facts
        cand["covered_security_facts"] = covered_sec_facts
        cand["num_security_facts"] = int(num_sec_facts)
        cand["f_sec"] = float(f_sec)

        cand["hallucination_penalty"] = float(hallucination_penalty)
        cand["hallucination_flags"] = hallucination_flags

        cand["f_style"] = float(f_style)
        cand["style_flags"] = style_flags

        cand["score"] = float(final_score)

    # ------------------------------------------------------------
    # 4. Sort candidates and save
    # ------------------------------------------------------------

    for record in records:
        candidates = record.get("candidates", [])

        # 只保留有 score 的候选
        candidates = [c for c in candidates if "score" in c]

        candidates = sorted(
            candidates,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )

        record["candidates"] = candidates
        record["num_candidates"] = len(candidates)

        if candidates:
            record["best_candidate"] = candidates[0]
            record["worst_candidate"] = candidates[-1]
            record["score_range"] = float(candidates[0]["score"] - candidates[-1]["score"])
        else:
            record["best_candidate"] = None
            record["worst_candidate"] = None
            record["score_range"] = 0.0

    save_jsonl(records, args.output_file)

    # ------------------------------------------------------------
    # 5. Save summary
    # ------------------------------------------------------------

    summary = {
        "candidate_file": args.candidate_file,
        "output_file": args.output_file,
        "num_records": len(records),
        "num_candidates": len(flat),
        "weights": {
            "w_code": args.w_code,
            "w_sem": args.w_sem,
            "w_sec": args.w_sec,
            "w_style": args.w_style,
            "w_hall": args.w_hall,
        },
        "side_model_path": args.side_model_path,
        "sem_model_path": args.sem_model_path,
    }

    summary_path = str(Path(args.output_file).with_suffix(".summary.json"))

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Summary] Saved to {summary_path}")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate_file",
        type=str,
        default="data/dpo/candidate_comments.jsonl",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/dpo/scored_candidates.jsonl",
    )

    parser.add_argument(
        "--side_model_path",
        type=str,
        default="/home/data/wb/commentGeneration/SIRCOT/model/hard-negatives/141205/",
    )

    parser.add_argument(
        "--sem_model_path",
        type=str,
        default="/home/data/wb/commentGeneration/SIRCOT/model/all-MiniLM-L6-v2/",
        help=(
            "SentenceTransformer 模型路径。"
            "如果服务器不能联网，请改成本地 Sentence-BERT 模型路径。"
        )
    )

    parser.add_argument("--side_device", type=str, default="cpu")
    parser.add_argument("--sem_device", type=str, default="cpu")

    parser.add_argument("--side_batch_size", type=int, default=32)
    parser.add_argument("--sem_batch_size", type=int, default=32)

    parser.add_argument("--side_max_length", type=int, default=256)

    # final score weights
    parser.add_argument("--w_code", type=float, default=0.40)
    parser.add_argument("--w_sem", type=float, default=0.20)
    parser.add_argument("--w_sec", type=float, default=0.25)
    parser.add_argument("--w_style", type=float, default=0.10)
    parser.add_argument("--w_hall", type=float, default=0.15)

    args = parser.parse_args()

    print("=" * 80)
    print("PrefSCom: Candidate Scoring")
    print("=" * 80)
    print(f"Candidate file : {args.candidate_file}")
    print(f"Output file    : {args.output_file}")
    print(f"SIDE model     : {args.side_model_path}")
    print(f"SEM model      : {args.sem_model_path}")
    print(f"SIDE device    : {args.side_device}")
    print(f"SEM device     : {args.sem_device}")
    print(
        f"Weights        : "
        f"code={args.w_code}, sem={args.w_sem}, sec={args.w_sec}, "
        f"style={args.w_style}, hall={args.w_hall}"
    )
    print("=" * 80)

    score_candidates(args)


if __name__ == "__main__":
    main()

