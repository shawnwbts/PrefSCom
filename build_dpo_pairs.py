
"""
PrefSCom: Build DPO Pairs

功能：
1. 读取 data/dpo/scored_candidates.jsonl
2. 根据候选注释综合分数构造 chosen/rejected pairs
3. 确保：
   - hard_negative / negative_only 不会作为 chosen
   - reference / positive_anchor 不会作为 rejected
   - 每个函数最多使用有限数量的 hard-negative pair
4. 保存为 data/dpo/dpo_pairs.jsonl
5. 输出 DPO 数据统计信息

推荐运行：
python build_dpo_pairs.py \
  --scored_file data/dpo/scored_candidates.jsonl \
  --output_file data/dpo/dpo_pairs_clean.jsonl \
  --min_margin 0.20 \
  --min_chosen_score 0.70 \
  --max_rejected_score 0.55 \
  --max_pairs_per_sample 3 \
  --max_hard_negative_pairs 1
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


SYSTEM_PROMPT = (
    "You are an expert Solidity developer. "
    "Generate one short human-written style comment for the given Solidity function. "
    "The comment should be concise, natural, and faithful to the function behavior. "
    "Use 5 to 15 words when possible. "
    "Mention security-related constraints only when they are explicitly present in the function. "
    "Do not infer ownership, permissions, return values, emergency behavior, "
    "or safety guarantees unless they are explicitly shown in the function."
)


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
                data.append(json.loads(line))
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

    print(f"[Data] Saved {len(data)} DPO pairs to {path}")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def normalize_display_text(text: str) -> str:
    """
    用于写入 DPO pair 的轻量清洗。
    避免把 //、@dev、markdown 等格式污染写入 chosen/rejected。
    """
    text = str(text or "").strip()

    text = text.replace("/**", " ")
    text = text.replace("/*", " ")
    text = text.replace("*/", " ")
    text = text.replace("///", " ")
    text = text.replace("//", " ")
    text = text.replace("```solidity", " ")
    text = text.replace("```", " ")

    text = re.sub(r"^\s*\*\s*", "", text)
    text = re.sub(r"^\s*@dev\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*@notice\s+", "", text, flags=re.IGNORECASE)

    # 如果包含 @param/@return，截断到这些标签之前
    text = re.split(r"\s+@(param|return|returns)\b", text, flags=re.IGNORECASE)[0].strip()

    text = " ".join(text.split()).strip()

    return text


def build_prompt(instruction: str, function_code: str) -> str:
    """
    保存纯文本 prompt。
    后续 train_dpo_model.py 中可以再用 tokenizer.apply_chat_template 包装成 Qwen chat format。
    """
    return (
        f"{instruction}\n\n"
        f"Solidity function:\n"
        f"{function_code}\n\n"
        "Generate exactly one short comment. "
        "Use 5 to 15 words when possible. "
        "Describe only the main behavior of the function. "
        "Do not include code, markdown, explanations, return descriptions, "
        "ownership claims, emergency behavior, or security guarantees unless they are explicitly supported by the code."
    )


# ============================================================
# 2. Candidate Type Judgement
# ============================================================

def get_source(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("source", "") or "")


def get_role_hint(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("role_hint", "") or "")


def is_hard_negative(candidate: Dict[str, Any]) -> bool:
    source = get_source(candidate)
    role_hint = get_role_hint(candidate)
    return source.startswith("hard_negative") or role_hint == "negative_only"


def is_reference(candidate: Dict[str, Any]) -> bool:
    return get_source(candidate) == "reference"


def is_positive_anchor(candidate: Dict[str, Any]) -> bool:
    return get_role_hint(candidate) == "positive_anchor"


def has_hallucination(candidate: Dict[str, Any]) -> bool:
    flags = candidate.get("hallucination_flags", [])
    return isinstance(flags, list) and len(flags) > 0


# ============================================================
# 3. Comment Quality Filters
# ============================================================

def is_probably_english(text: str) -> bool:
    text = str(text or "").strip()

    if not text:
        return False

    letters = sum(c.isalpha() for c in text)
    ascii_letters = sum(("a" <= c.lower() <= "z") for c in text)

    if letters == 0:
        return False

    lower = text.lower()

    non_english_markers = [
        "functie", "terug", "geeft",
        "fonction", "retourne",
        "función", "devuelve",
        "funktion", "gibt",
        "funzione", "restituisce",
    ]

    if any(x in lower for x in non_english_markers):
        return False

    return ascii_letters / letters >= 0.95


def has_comment_artifact(text: str) -> bool:
    lower = str(text or "").lower()

    bad_tokens = [
        "/**", "/*", "*/", "///", "//",
        "@param", "@return", "@returns", "@dev", "@notice",
        "```"
    ]

    return any(tok in lower for tok in bad_tokens)


def has_return_artifact(text: str) -> bool:
    """
    检测模型把 @return 描述混进一句话注释的情况。
    合法的 'Returns the balance.' 不应该被惩罚。
    """
    lower = " " + normalize_text(text) + " "

    bad_patterns = [
        " return ",
        " return the ",
        " return a ",
        " return an ",
        " returns uint",
        " return uint",
        " returns bool",
        " return bool",
        " returns string",
        " return string",
        " returns address",
        " return address",
        " representing the ",
    ]

    return any(p in lower for p in bad_patterns)


def is_too_generic_comment(text: str) -> bool:
    lower = normalize_text(text).strip().rstrip(".")

    generic_exact = {
        "executes the function",
        "execute the function",
        "performs the operation",
        "perform the operation",
        "performs the requested operation",
        "does something",
        "main function",
        "internal function",
        "external function",
    }

    return lower in generic_exact


def is_valid_comment(comment: str) -> bool:
    text = normalize_display_text(comment)
    lower = text.lower()

    if not text:
        return False

    words = text.split()

    # 过短没有训练价值
    if len(words) < 3:
        return False

    # 过长容易导致模型学习冗长注释
    if len(words) > 40:
        return False

    # 过滤明显格式污染
    if has_comment_artifact(comment):
        return False

    # 过滤明显非英文
    if not is_probably_english(text):
        return False

    # 过滤 return 描述残留
    if has_return_artifact(text):
        return False

    # 过滤泛化废话
    if is_too_generic_comment(text):
        return False

    # 过滤明显模板化输出
    bad_phrases = [
        "here is",
        "the comment is",
        "generated comment",
        "as an ai",
        "this function is used to",
    ]

    if any(p in lower for p in bad_phrases):
        return False

    # 过滤明显重复输出
    if len(words) >= 8:
        first_half = " ".join(words[:len(words) // 2]).lower()
        second_half = " ".join(words[len(words) // 2:]).lower()
        if first_half and first_half == second_half:
            return False

    return True


# ============================================================
# 4. Pair Score Adjustment
# ============================================================

def length_penalty_for_pair_selection(comment: str) -> float:
    """
    用于 DPO pair 选择阶段的轻量长度惩罚。
    不改变 scored_candidates.jsonl 中的原始 score，只影响 pair 选择。
    """
    words = normalize_display_text(comment).split()
    n = len(words)

    if 5 <= n <= 15:
        return 0.0
    elif n < 5:
        return 0.03
    elif n <= 25:
        return 0.05
    elif n <= 40:
        return 0.10
    else:
        return 0.20


def adjusted_pair_score(candidate: Dict[str, Any]) -> float:
    raw_score = float(candidate.get("score", 0.0))
    penalty = length_penalty_for_pair_selection(candidate.get("comment", ""))
    return raw_score - penalty


# ============================================================
# 5. Chosen / Rejected Candidate Filters
# ============================================================

def is_valid_chosen_candidate(
    candidate: Dict[str, Any],
    min_chosen_score: float,
    min_chosen_style: float,
) -> bool:
    """
    chosen 必须是 reference 或模型生成的高质量候选。
    hard negative / negative_only 绝不能作为 chosen。
    """
    if is_hard_negative(candidate):
        return False

    if not is_valid_comment(candidate.get("comment", "")):
        return False

    score = float(candidate.get("score", 0.0))
    if score < min_chosen_score:
        return False

    # 有明显幻觉的候选不作为 chosen
    if float(candidate.get("hallucination_penalty", 0.0)) > 0:
        return False

    # 风格太差的不作为 chosen
    f_style = candidate.get("f_style", None)
    if f_style is not None and float(f_style) < min_chosen_style:
        return False

    return True


def is_valid_rejected_candidate(
    candidate: Dict[str, Any],
    max_rejected_score: float,
) -> bool:
    """
    rejected 可以是 hard negative 或低分模型候选。
    reference / positive_anchor 不应该作为 rejected。
    """
    if is_reference(candidate):
        return False

    if is_positive_anchor(candidate):
        return False

    if not is_valid_comment(candidate.get("comment", "")):
        return False

    score = float(candidate.get("score", 0.0))
    if score > max_rejected_score:
        return False

    return True


# ============================================================
# 6. Pair Construction
# ============================================================

def make_pair(
    record: Dict[str, Any],
    prompt: str,
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    pair_type: str,
    pair_index: int,
) -> Dict[str, Any]:
    chosen_comment = normalize_display_text(chosen.get("comment", ""))
    rejected_comment = normalize_display_text(rejected.get("comment", ""))

    chosen_score = float(chosen.get("score", 0.0))
    rejected_score = float(rejected.get("score", 0.0))

    chosen_adjusted_score = adjusted_pair_score(chosen)
    rejected_adjusted_score = adjusted_pair_score(rejected)

    return {
        "id": f"{record.get('id')}_pair_{pair_index}",
        "source_id": record.get("id"),
        "prompt": prompt,
        "system_prompt": SYSTEM_PROMPT,
        "function": record.get("function", ""),
        "reference": record.get("reference", ""),

        "chosen": chosen_comment,
        "rejected": rejected_comment,

        "chosen_source": chosen.get("source", "unknown"),
        "rejected_source": rejected.get("source", "unknown"),

        "chosen_role_hint": chosen.get("role_hint", ""),
        "rejected_role_hint": rejected.get("role_hint", ""),

        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "chosen_adjusted_score": chosen_adjusted_score,
        "rejected_adjusted_score": rejected_adjusted_score,
        "margin": chosen_adjusted_score - rejected_adjusted_score,
        "pair_type": pair_type,

        "chosen_f_code": chosen.get("f_code"),
        "chosen_f_sem": chosen.get("f_sem"),
        "chosen_f_sec": chosen.get("f_sec"),
        "chosen_f_style": chosen.get("f_style"),
        "chosen_hallucination_penalty": chosen.get("hallucination_penalty"),

        "rejected_f_code": rejected.get("f_code"),
        "rejected_f_sem": rejected.get("f_sem"),
        "rejected_f_sec": rejected.get("f_sec"),
        "rejected_f_style": rejected.get("f_style"),
        "rejected_hallucination_penalty": rejected.get("hallucination_penalty"),

        "chosen_security_facts": chosen.get("covered_security_facts", []),
        "rejected_security_facts": rejected.get("covered_security_facts", []),
        "chosen_hallucination_flags": chosen.get("hallucination_flags", []),
        "rejected_hallucination_flags": rejected.get("hallucination_flags", []),
    }


def select_pairs_for_record(
    record: Dict[str, Any],
    min_margin: float,
    max_pairs_per_sample: int,
    min_chosen_score: float,
    max_rejected_score: float,
    min_chosen_style: float,
    prefer_hard_negative: bool = True,
    max_hard_negative_pairs: int = 1,
) -> List[Dict[str, Any]]:
    """
    为单个函数构造 DPO pairs。

    约束：
    1. hard_negative / negative_only 不能作为 chosen；
    2. reference / positive_anchor 不能作为 rejected；
    3. 每个样本最多使用 max_hard_negative_pairs 个 hard-negative pair；
    4. 其余 pair 优先使用低分模型候选作为 rejected。
    """

    candidates = record.get("candidates", [])
    if len(candidates) < 2:
        return []

    valid_candidates = []
    seen_comments = set()

    for c in candidates:
        if "score" not in c:
            continue

        comment = c.get("comment", "")
        if not is_valid_comment(comment):
            continue

        key = normalize_text(normalize_display_text(comment))
        if key in seen_comments:
            continue

        seen_comments.add(key)
        valid_candidates.append(c)

    if len(valid_candidates) < 2:
        return []

    chosen_pool = [
        c for c in valid_candidates
        if is_valid_chosen_candidate(
            candidate=c,
            min_chosen_score=min_chosen_score,
            min_chosen_style=min_chosen_style,
        )
    ]

    rejected_pool = [
        c for c in valid_candidates
        if is_valid_rejected_candidate(
            candidate=c,
            max_rejected_score=max_rejected_score,
        )
    ]

    if not chosen_pool or not rejected_pool:
        return []

    chosen_pool = sorted(
        chosen_pool,
        key=lambda x: adjusted_pair_score(x),
        reverse=True
    )

    rejected_pool = sorted(
        rejected_pool,
        key=lambda x: adjusted_pair_score(x)
    )

    hard_negative_pool = [
        c for c in rejected_pool
        if is_hard_negative(c)
    ]

    model_negative_pool = [
        c for c in rejected_pool
        if not is_hard_negative(c)
    ]

    prompt = build_prompt(
        instruction=record.get(
            "instruction",
            "Generate a concise and accurate comment for the given Solidity function."
        ),
        function_code=record.get("function", "")
    )

    pairs = []
    used_pair_keys = set()
    hard_negative_pair_count = 0

    def try_add_pair(chosen, rejected, pair_type: str):
        nonlocal hard_negative_pair_count

        if len(pairs) >= max_pairs_per_sample:
            return

        # 再做一次硬约束
        if not is_valid_chosen_candidate(chosen, min_chosen_score, min_chosen_style):
            return

        if not is_valid_rejected_candidate(rejected, max_rejected_score):
            return

        if is_hard_negative(rejected):
            if hard_negative_pair_count >= max_hard_negative_pairs:
                return

        chosen_comment = normalize_display_text(chosen.get("comment", ""))
        rejected_comment = normalize_display_text(rejected.get("comment", ""))

        if normalize_text(chosen_comment) == normalize_text(rejected_comment):
            return

        chosen_adjusted_score = adjusted_pair_score(chosen)
        rejected_adjusted_score = adjusted_pair_score(rejected)
        margin = chosen_adjusted_score - rejected_adjusted_score

        if margin < min_margin:
            return

        pair_key = (normalize_text(chosen_comment), normalize_text(rejected_comment))
        if pair_key in used_pair_keys:
            return

        used_pair_keys.add(pair_key)

        if is_hard_negative(rejected):
            hard_negative_pair_count += 1

        pair = make_pair(
            record=record,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            pair_type=pair_type,
            pair_index=len(pairs),
        )

        pairs.append(pair)

    # 1. 最优 chosen
    best_chosen = chosen_pool[0]

    # 2. 优先构造有限数量 hard-negative pair
    if prefer_hard_negative and hard_negative_pool:
        for hn in hard_negative_pool:
            try_add_pair(best_chosen, hn, pair_type="top_vs_hard_negative")
            if hard_negative_pair_count >= max_hard_negative_pairs:
                break

    # 3. 再构造 top chosen vs 低分模型候选
    for idx, rejected in enumerate(model_negative_pool[:5]):
        try_add_pair(best_chosen, rejected, pair_type=f"top_vs_low_model_{idx + 1}")
        if len(pairs) >= max_pairs_per_sample:
            return pairs

    # 4. 如果还有空间，用第二好的 chosen 配低分 rejected
    if len(chosen_pool) >= 2:
        second_chosen = chosen_pool[1]

        for idx, rejected in enumerate(rejected_pool[:5]):
            try_add_pair(second_chosen, rejected, pair_type=f"second_top_vs_low_{idx + 1}")
            if len(pairs) >= max_pairs_per_sample:
                return pairs

    return pairs[:max_pairs_per_sample]


# ============================================================
# 7. Main Build Pipeline
# ============================================================

def build_dpo_pairs(args):
    records = load_jsonl(args.scored_file)

    all_pairs = []
    skipped_no_pairs = 0

    for record in records:
        pairs = select_pairs_for_record(
            record=record,
            min_margin=args.min_margin,
            max_pairs_per_sample=args.max_pairs_per_sample,
            min_chosen_score=args.min_chosen_score,
            max_rejected_score=args.max_rejected_score,
            min_chosen_style=args.min_chosen_style,
            prefer_hard_negative=args.prefer_hard_negative,
            max_hard_negative_pairs=args.max_hard_negative_pairs,
        )

        if not pairs:
            skipped_no_pairs += 1
            continue

        all_pairs.extend(pairs)

    save_jsonl(all_pairs, args.output_file)

    # 统计信息
    pair_type_counter = Counter()
    chosen_source_counter = Counter()
    rejected_source_counter = Counter()
    chosen_role_counter = Counter()
    rejected_role_counter = Counter()
    margin_values = []

    invalid_chosen_hard_negative = 0
    invalid_rejected_reference = 0

    for p in all_pairs:
        pair_type_counter[p["pair_type"]] += 1
        chosen_source_counter[p["chosen_source"]] += 1
        rejected_source_counter[p["rejected_source"]] += 1
        chosen_role_counter[p.get("chosen_role_hint", "")] += 1
        rejected_role_counter[p.get("rejected_role_hint", "")] += 1
        margin_values.append(p["margin"])

        if str(p["chosen_source"]).startswith("hard_negative") or p.get("chosen_role_hint") == "negative_only":
            invalid_chosen_hard_negative += 1

        if p["rejected_source"] == "reference" or p.get("rejected_role_hint") == "positive_anchor":
            invalid_rejected_reference += 1

    summary = {
        "scored_file": args.scored_file,
        "output_file": args.output_file,
        "num_records": len(records),
        "num_dpo_pairs": len(all_pairs),
        "skipped_records_without_pairs": skipped_no_pairs,

        "min_margin": args.min_margin,
        "min_chosen_score": args.min_chosen_score,
        "max_rejected_score": args.max_rejected_score,
        "min_chosen_style": args.min_chosen_style,
        "max_pairs_per_sample": args.max_pairs_per_sample,
        "prefer_hard_negative": args.prefer_hard_negative,
        "max_hard_negative_pairs": args.max_hard_negative_pairs,

        "pair_type_distribution": dict(pair_type_counter),
        "chosen_source_distribution": dict(chosen_source_counter),
        "rejected_source_distribution": dict(rejected_source_counter),
        "chosen_role_distribution": dict(chosen_role_counter),
        "rejected_role_distribution": dict(rejected_role_counter),

        "avg_margin": sum(margin_values) / len(margin_values) if margin_values else 0.0,
        "max_margin": max(margin_values) if margin_values else 0.0,
        "min_margin_actual": min(margin_values) if margin_values else 0.0,

        "invalid_chosen_hard_negative": invalid_chosen_hard_negative,
        "invalid_rejected_reference": invalid_rejected_reference,
    }

    summary_file = str(Path(args.output_file).with_suffix(".summary.json"))

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DPO Pair Construction Summary")
    print("=" * 80)
    print(f"Records: {len(records)}")
    print(f"DPO pairs: {len(all_pairs)}")
    print(f"Skipped records without valid pairs: {skipped_no_pairs}")
    print(f"Avg margin: {summary['avg_margin']:.4f}")
    print(f"Invalid chosen hard-negative: {invalid_chosen_hard_negative}")
    print(f"Invalid rejected reference: {invalid_rejected_reference}")
    print(f"Pair types: {dict(pair_type_counter)}")
    print(f"Chosen sources: {dict(chosen_source_counter)}")
    print(f"Rejected sources: {dict(rejected_source_counter)}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)


# ============================================================
# 8. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scored_file",
        type=str,
        default="data/dpo/scored_candidates.jsonl"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/dpo/dpo_pairs.jsonl"
    )

    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.20,
        help="Only construct pairs with adjusted chosen_score - rejected_score >= min_margin."
    )

    parser.add_argument(
        "--min_chosen_score",
        type=float,
        default=0.70,
        help="Minimum raw score for chosen candidates."
    )

    parser.add_argument(
        "--max_rejected_score",
        type=float,
        default=0.55,
        help="Maximum raw score for rejected candidates."
    )

    parser.add_argument(
        "--min_chosen_style",
        type=float,
        default=0.60,
        help="Minimum f_style for chosen candidates."
    )

    parser.add_argument(
        "--max_pairs_per_sample",
        type=int,
        default=3,
        help="Maximum DPO pairs generated from each function."
    )

    parser.add_argument(
        "--prefer_hard_negative",
        action="store_true",
        default=True,
        help="Prefer using rule-based hard negatives as rejected candidates."
    )

    parser.add_argument(
        "--max_hard_negative_pairs",
        type=int,
        default=1,
        help="Maximum hard-negative pairs generated from each function."
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PrefSCom: Build DPO Pairs")
    print("=" * 80)
    print(f"Scored file             : {args.scored_file}")
    print(f"Output file             : {args.output_file}")
    print(f"Min margin              : {args.min_margin}")
    print(f"Min chosen score        : {args.min_chosen_score}")
    print(f"Max rejected score      : {args.max_rejected_score}")
    print(f"Min chosen style        : {args.min_chosen_style}")
    print(f"Max pairs per sample    : {args.max_pairs_per_sample}")
    print(f"Prefer hard negative    : {args.prefer_hard_negative}")
    print(f"Max hard negative pairs : {args.max_hard_negative_pairs}")
    print("=" * 80)

    build_dpo_pairs(args)


if __name__ == "__main__":
    main()