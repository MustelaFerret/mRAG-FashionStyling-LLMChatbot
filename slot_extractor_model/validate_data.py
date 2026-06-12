"""Validation gate (per the project rule: never trust GenAI data blind). A blind judge
(claude-sonnet-4.5) checks each claimed span: does the substring REALLY express that
attribute in that query, and is the canonical mapping right? It also flags any attribute
present but NOT labelled (under-labelling). Rows with any bad/missing span are dropped.
Substring-presence alone is insufficient — the generator fabricates spans (e.g. tagging the
garment "top" as a season).

Judges in BATCHES (~10 queries/request) to cut the per-request cost floor ~10x; falls back
to per-row if a batch reply is malformed. Aborts on HTTP 402 (out of balance) instead of
dropping every row as judge-fail.

Run: python -m slot_extractor_model.validate_data --in data/train_raw.csv --out data/train_clean.csv
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from slot_extractor_model import config as C

BATCH = 10


class BillingError(RuntimeError):
    """API returned 402 (out of balance) — abort instead of dropping every row as judge-fail."""


FIELD_DESC = {
    "colour_group": "a colour", "fit": "a fit/silhouette", "occasion": "an occasion/use",
    "seasonality": "a season",
}


def _claims_of(row) -> dict:
    return {f: (str(row[f + "_surface"]).strip(), row[f + "_canon"])
            for f in C.FIELDS if row[f + "_canon"] != "none"}


def _item_block(i: int, query: str, claims: dict) -> str:
    lines = [f"  - {f} = \"{s}\" (should mean {FIELD_DESC[f]}: {c})" for f, (s, c) in claims.items()]
    return f"#{i} query: \"{query}\"\n" + "\n".join(lines)


def _post(messages, max_tokens, retries=5):
    payload = {"model": C.JUDGE_MODEL, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {C.API_KEY}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            r = requests.post(f"{C.API_BASE}/chat/completions", headers=headers, json=payload, timeout=180)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code == 402:
                raise BillingError(r.text.encode("ascii", "backslashreplace").decode()[:200])
            time.sleep(2 * (attempt + 1))
        except BillingError:
            raise
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


SYS = "You are a strict annotation validator. Output ONLY JSON."


def _judge_one(query: str, claims: dict):
    prompt = (
        f"Query: \"{query}\"\nClaimed attribute spans (substring -> canonical):\n"
        + "\n".join(f"- {f} = \"{s}\" (should mean {FIELD_DESC[f]}: {c})" for f, (s, c) in claims.items())
        + "\n\nFor EACH claim decide if the substring genuinely expresses that attribute (a colour word "
        "like 'oxblood' counts as colour; a garment word like 'top' does NOT count as a season). Also "
        "list any OTHER colour/fit/occasion/season clearly present but not claimed.\n"
        'Return ONLY JSON: {"ok": {"<field>": true/false}, "extra": ["<field>"]}'
    )
    txt = _post([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}], 300)
    if not txt:
        return None
    try:
        s, e = txt.find("{"), txt.rfind("}")
        return json.loads(txt[s:e + 1])
    except Exception:
        return None


def _judge_batch(items):
    """items: list of (i, query, claims). Returns dict i->verdict, or None on malformed reply."""
    blocks = "\n\n".join(_item_block(i, q, c) for i, q, c in items)
    prompt = (
        f"Validate {len(items)} annotations. For EACH, decide per claim if the substring genuinely "
        "expresses that attribute (a colour word like 'oxblood' counts as colour; a garment word like "
        "'top' does NOT count as a season), and list any OTHER colour/fit/occasion/season present but "
        "not claimed.\n\n" + blocks +
        '\n\nReturn ONLY a JSON array, one object per item IN ORDER: '
        '[{"i": <number>, "ok": {"<field>": true/false}, "extra": ["<field>"]}]'
    )
    txt = _post([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}], 1500)
    if not txt:
        return None
    try:
        s, e = txt.find("["), txt.rfind("]")
        arr = json.loads(txt[s:e + 1])
    except Exception:
        return None
    by_i = {int(o["i"]): o for o in arr if isinstance(o, dict) and "i" in o}
    if len(by_i) != len(items):
        return None
    return by_i


def _verdict_bad(verdict, claims):
    bad = [f for f in claims if not verdict.get("ok", {}).get(f, False)]
    extra = verdict.get("extra", []) or []
    return bad or extra, f"bad={bad} extra={extra}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/train_raw.csv")
    ap.add_argument("--out", default="data/train_clean.csv")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    df = pd.read_csv(C.ROOT / a.inp).fillna("")

    rows = [r for _, r in df.iterrows()]
    kept, dropped, drop_reasons = [], 0, []

    # auto-keep rows with no claims (all-none) + only-english
    to_judge = []
    for r in rows:
        if not C.is_english(r["query"]):
            dropped += 1; drop_reasons.append((r["query"], "non-english")); continue
        claims = _claims_of(r)
        if not claims:
            kept.append(r)
        else:
            to_judge.append((r, claims))

    batches = [to_judge[i:i + BATCH] for i in range(0, len(to_judge), BATCH)]

    def _do_batch(batch):
        items = [(j, r["query"], c) for j, (r, c) in enumerate(batch)]
        res = _judge_batch(items)
        out = []
        if res is None:  # fallback per-row
            for r, c in batch:
                v = _judge_one(r["query"], c)
                out.append((r, c, v))
        else:
            for j, (r, c) in enumerate(batch):
                out.append((r, c, res.get(j)))
        return out

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for batch_out in ex.map(_do_batch, batches):
                for r, c, v in batch_out:
                    if v is None:
                        dropped += 1; drop_reasons.append((r["query"], "judge-fail")); continue
                    is_bad, why = _verdict_bad(v, c)
                    if is_bad:
                        dropped += 1; drop_reasons.append((r["query"], why))
                    else:
                        kept.append(r)
    except BillingError as e:
        raise SystemExit(f"[validate] ABORTED — API out of balance (402): {e}\n"
                         f"  Top up ckey.vn then re-run; raw ({a.inp}) intact, NOTHING written.")

    pd.DataFrame(kept).to_csv(C.ROOT / a.out, index=False, encoding="utf-8")
    print(f"[validate] kept {len(kept)} / dropped {dropped} of {len(df)} -> {a.out}")
    for q, why in drop_reasons[:10]:
        print(f"    DROP [{why}] {q}")


if __name__ == "__main__":
    main()
