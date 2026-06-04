from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import pandas as pd
import requests
from tqdm.auto import tqdm

from intent_analysis_model import config as C


class IntentJudge:
    def __init__(self, model: str, batch_size: int = 20, max_retries: int = 4):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        if not C.API_KEY:
            raise RuntimeError("CKEY_API_KEY rỗng — kiểm tra .env")

    def _user_prompt(self, batch: List[str]) -> str:
        defs = "\n".join(f"- {k}: {v}" for k, v in C.INTENT_DEFINITIONS.items())
        rules = "\n".join(f"- {r}" for r in C.HARD_CASE_RULES)
        labels = ", ".join(f'"{l}"' for l in C.INTENT_LABELS)
        lines = "\n".join(f"{i}: {q}" for i, q in enumerate(batch))
        return (
            "You are a strict labeler for a fashion shopping assistant's intent classifier. "
            "Classify each query into EXACTLY ONE intent. Query length must NOT influence the label.\n\n"
            f"Intent definitions:\n{defs}\n\n"
            f"Boundary rules:\n{rules}\n\n"
            f"You MUST use ONLY these exact label strings (no other names): {labels}.\n\n"
            "Output ONLY a JSON array, one object per query, using EXACTLY these keys:\n"
            '[{"i": 0, "intent": "<one of the exact labels>"}, {"i": 1, "intent": "..."}]\n\n'
            f"Queries:\n{lines}"
        )

    def _call(self, batch: List[str]) -> Dict[int, str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self._user_prompt(batch)},
            ],
            "temperature": 0,
            "max_tokens": 2500,
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
        return {}

    @staticmethod
    def _parse(content: str) -> Dict[int, str]:
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            content = parts[1].replace("json", "", 1).strip() if len(parts) >= 2 else content
        s, e = content.find("["), content.rfind("]")
        if s == -1 or e == -1:
            return {}
        try:
            arr = json.loads(content[s : e + 1])
        except Exception:
            return {}
        out = {}
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            idx = obj.get("i", obj.get("index"))
            lab = obj.get("intent", obj.get("label"))
            if idx is None or lab is None:
                continue
            try:
                out[int(idx)] = str(lab).strip()
            except Exception:
                continue
        return out

    def judge(self, df: pd.DataFrame) -> pd.DataFrame:
        queries = df["query"].tolist()
        judged: List[str] = [""] * len(queries)
        for start in tqdm(range(0, len(queries), self.batch_size), desc=f"judge[{self.model}]"):
            batch = queries[start : start + self.batch_size]
            res = self._call(batch)
            for local_i, label in res.items():
                if 0 <= local_i < len(batch):
                    judged[start + local_i] = label
        out = df.copy()
        out["judge_intent"] = judged
        return out


def _report(df: pd.DataFrame, name: str):
    total = len(df)
    judged_ok = (df["judge_intent"] != "").sum()
    kept = (df["judge_intent"] == df["intent"]).sum()
    print(f"\n[{name}] total={total} judged={judged_ok} kept(match)={kept} ({kept/total*100:.1f}%)")
    print("  pass-rate per intent:")
    for intent in C.INTENT_LABELS:
        sub = df[df["intent"] == intent]
        if len(sub):
            k = (sub["judge_intent"] == intent).sum()
            print(f"    {intent:18s} {k}/{len(sub)} ({k/len(sub)*100:.0f}%)")
    mism = df[(df["judge_intent"] != "") & (df["judge_intent"] != df["intent"])]
    if len(mism):
        print("  top mismatch (gen->judge):")
        for (g, j), c in mism.groupby(["intent", "judge_intent"]).size().sort_values(ascending=False).head(8).items():
            print(f"    {g} -> {j}: {c}")


def validate_file(raw_path, judge_model: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    before = len(df)
    df = df[df["query"].apply(C.is_english)].reset_index(drop=True)
    if len(df) < before:
        print(f"[{name}] lọc English-only: bỏ {before - len(df)} câu tiếng Việt ({before}->{len(df)})")
    judge = IntentJudge(model=judge_model)
    judged = judge.judge(df)
    _report(judged, name)
    kept = judged[judged["judge_intent"] == judged["intent"]].copy()
    kept = kept[["query", "intent"]].reset_index(drop=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = ap.parse_args()

    if args.split in ("train", "both"):
        kept = validate_file(C.DATA_DIR / "raw_train.csv", C.JUDGE_MODEL_TRAIN, "train")
        val_rows, train_rows = [], []
        for intent, sub in kept.groupby("intent"):
            sub = sub.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n_val = int(len(sub) * C.VAL_RATIO)
            val_rows.append(sub.iloc[:n_val])
            train_rows.append(sub.iloc[n_val:])
        train_df = pd.concat(train_rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
        val_df = pd.concat(val_rows).sample(frac=1.0, random_state=2).reset_index(drop=True)
        train_df.to_csv(C.DATA_DIR / "train.csv", index=False)
        val_df.to_csv(C.DATA_DIR / "val.csv", index=False)
        print(f"\nsaved train.csv ({len(train_df)}) + val.csv ({len(val_df)})")

    if args.split in ("test", "both"):
        kept = validate_file(C.DATA_DIR / "raw_test.csv", C.JUDGE_MODEL_TEST, "test")
        kept.to_csv(C.DATA_DIR / "test.csv", index=False)
        print(f"\nsaved test.csv ({len(kept)})")


if __name__ == "__main__":
    main()
