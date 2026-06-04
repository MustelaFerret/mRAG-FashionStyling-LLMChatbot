"""Sinh data intent classification qua ckey.vn — multi-model, stratified theo độ dài.

Nguyên tắc chống length bias:
- Mỗi intent sinh ĐỀU 3 dải độ dài (short/medium/long). similar/graph/color BẮT BUỘC có
  nhiều câu DÀI chi tiết → model không thể dùng độ dài làm shortcut cho composite.
- Train xoay tua nhiều model; test dùng model pool KHÁC (held-out style).
- Prompt nhúng định nghĩa CHẶT + hard contrastive rules.

Run:
    conda activate mRAG
    python -m intent_analysis_model.gen_data            # cả train + test
    python -m intent_analysis_model.gen_data --split test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from typing import Dict, List

import pandas as pd
import requests
from tqdm.auto import tqdm

from intent_analysis_model import config as C


FASHION_CONTEXTS = [
    "dresses", "jackets & coats", "denim & jeans", "knitwear & sweaters", "shoes & boots",
    "bags & accessories", "activewear", "formalwear & tailoring", "streetwear", "lingerie & basics",
    "skirts", "trousers", "tops & blouses", "hoodies & sweatshirts", "jewelry", "swimwear",
    "hats & beanies", "scarves", "loungewear", "vintage & retro pieces",
]

BUCKET_GUIDANCE = {
    "short": "Very terse — keyword style, no full sentences. e.g. 'red midi dress', 'denim jacket'.",
    "medium": "A short natural sentence or phrase with 1-2 attributes.",
    "long": "A detailed, descriptive request with several attributes (color, material, fit, silhouette, "
            "occasion, era). Long but STILL one coherent intent — do not bolt on a second request.",
}


class DataGenerator:
    def __init__(self, batch_size: int = 25, temperature: float = 1.0, max_retries: int = 4):
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_retries = max_retries
        if not C.API_KEY:
            raise RuntimeError("CKEY_API_KEY rỗng — kiểm tra .env")

    def _system_prompt(self) -> str:
        defs = "\n".join(f"- {k}: {v}" for k, v in C.INTENT_DEFINITIONS.items())
        rules = "\n".join(f"- {r}" for r in C.HARD_CASE_RULES)
        return (
            "You are a precise data synthesizer for a fashion shopping assistant's intent classifier.\n"
            "There are exactly 5 intents:\n"
            f"{defs}\n\n"
            "CRITICAL boundary rules (avoid these classic mistakes):\n"
            f"{rules}\n\n"
            "Generate realistic queries a real shopper would type (typos, slang, lowercase allowed). "
            "Output ONLY a JSON array of strings, no commentary."
        )

    def _user_prompt(self, intent: str, bucket: str, n: int, context: str) -> str:
        b = C.LENGTH_BUCKETS[bucket]
        return (
            f"Generate {n} DIVERSE user queries that are STRICTLY and ONLY the intent '{intent}'.\n"
            f"Theme to draw from (for variety, not a hard constraint): {context}.\n"
            f"Length requirement: each query MUST be {b['min_words']}-{b['max_words']} words. "
            f"{BUCKET_GUIDANCE[bucket]}\n"
            f"Definition of '{intent}': {C.INTENT_DEFINITIONS[intent]}\n"
            "Every query must unambiguously belong to this intent and not be mistakable for another. "
            "Vary phrasing, vocabulary, and structure. Output ONLY a JSON array of strings."
        )

    def _call(self, model: str, intent: str, bucket: str, n: int, context: str) -> List[str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(intent, bucket, n, context)},
            ],
            "temperature": self.temperature,
            "max_tokens": 2000,
        }
        headers = {"Authorization": f"Bearer {C.API_KEY}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{C.API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
                if r.status_code != 200:
                    time.sleep(2 * (attempt + 1))
                    continue
                content = r.json()["choices"][0]["message"]["content"]
                return self._parse(content)
            except Exception:
                time.sleep(2 * (attempt + 1))
        return []

    @staticmethod
    def _parse(content: str) -> List[str]:
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1] if content.count("```") >= 2 else content
            content = content.replace("json", "", 1).strip()
        start, end = content.find("["), content.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            arr = json.loads(content[start : end + 1])
        except Exception:
            return []
        out = []
        for item in arr:
            if isinstance(item, str):
                q = item.strip()
            elif isinstance(item, dict):
                q = str(item.get("query", "")).strip()
            else:
                q = ""
            if q:
                out.append(q)
        return out

    def generate_split(self, per_intent: int, models: List[str], seed: int) -> pd.DataFrame:
        rng = random.Random(seed)
        rows: List[Dict] = []
        model_cycle = 0
        plan = []
        for intent in C.INTENT_LABELS:
            for bucket, spec in C.LENGTH_BUCKETS.items():
                target = int(round(per_intent * spec["ratio"]))
                n_calls = math.ceil(target / self.batch_size)
                for _ in range(n_calls):
                    plan.append((intent, bucket, min(self.batch_size, target)))
                    target -= self.batch_size
                    if target <= 0:
                        break

        for intent, bucket, n in tqdm(plan, desc="gen"):
            model = models[model_cycle % len(models)]
            model_cycle += 1
            context = rng.choice(FASHION_CONTEXTS)
            queries = self._call(model, intent, bucket, n, context)
            for q in queries:
                rows.append({"query": q, "intent": intent, "bucket": bucket, "gen_model": model})

        df = pd.DataFrame(rows)
        df["query"] = df["query"].str.strip()
        df = df[df["query"].str.len() > 0]
        df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--batch_size", type=int, default=25)
    args = ap.parse_args()

    gen = DataGenerator(batch_size=args.batch_size)

    if args.split in ("train", "both"):
        df = gen.generate_split(C.PER_INTENT_TRAIN, C.GEN_MODELS_TRAIN, seed=42)
        out = C.DATA_DIR / "raw_train.csv"
        df.to_csv(out, index=False)
        print(f"train raw: {len(df)} rows -> {out}")
        print(df["intent"].value_counts().to_string())
        print(df.groupby(["intent", "bucket"]).size().to_string())

    if args.split in ("test", "both"):
        df = gen.generate_split(C.PER_INTENT_TEST, C.GEN_MODELS_TEST, seed=7)
        out = C.DATA_DIR / "raw_test.csv"
        df.to_csv(out, index=False)
        print(f"test raw: {len(df)} rows -> {out}")
        print(df["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
