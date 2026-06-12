"""Label-conditioned synthetic data for the slot extractor. Sample a slot spec, ask an
LLM to WRITE shopper queries expressing EXACTLY that spec (using varied/rare surface forms),
label = the spec. Output ONLY a JSON array of strings.

Run a small probe first:
  python -m slot_extractor_model.gen_data --n_specs 12 --per 3 --model gpt-5.4-mini --out data/probe.csv
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import requests

from slot_extractor_model import config as C

_FIELD_PROB = 0.45  # each slot field independently present with this probability


def _sample_spec(rng: random.Random) -> Dict[str, Optional[str]]:
    garment = rng.choice(C.GARMENTS)
    spec: Dict[str, Optional[str]] = {f: None for f in C.FIELDS}
    for f in C.FIELDS:
        if f == "fit" and garment in C.NO_FIT_GARMENTS:
            continue  # fit is meaningless for footwear/accessories
        if rng.random() < _FIELD_PROB:
            pool = C.GEN_OCCASIONS if f == "occasion" else C.SLOT_VALUES[f]
            spec[f] = rng.choice(pool)
    # avoid all-empty specs being too frequent: force at least one field ~70% of the time
    if all(v is None for v in spec.values()) and rng.random() < 0.7:
        f = rng.choice([x for x in C.FIELDS if not (x == "fit" and garment in C.NO_FIT_GARMENTS)])
        pool = C.GEN_OCCASIONS if f == "occasion" else C.SLOT_VALUES[f]
        spec[f] = rng.choice(pool)
    spec["_garment"] = garment
    return spec


def _system_prompt() -> str:
    return (
        "You synthesize realistic fashion shopping queries for training a slot EXTRACTOR "
        "(sequence tagger). Given a target spec (a garment + a subset of attributes), write queries "
        "expressing EXACTLY those attributes and NO OTHER colour/fit/occasion/season. Real shoppers' "
        "phrasing: typos, slang, lowercase fine. Crucially VARY the colour wording — OFTEN use a "
        "natural synonym instead of the canonical name (e.g. 'burgundy' for Dark Red). For EACH query "
        "you MUST also report the exact substring (copied verbatim from the query) that expresses each "
        "specified attribute. The substring must be the SHORTEST head word(s) carrying the attribute — "
        "EXCLUDE prepositions/articles/fillers (for, to, a, the, with). E.g. occasion 'wedding' NOT "
        "'for a wedding'; 'gym' NOT 'to the gym'. Output ONLY a JSON array of objects: "
        '[{"q": "<query>", "spans": {"<field>": "<verbatim shortest substring>"}}].'
    )


def _user_prompt(spec: Dict, bucket: str, n: int) -> str:
    b = C.LENGTH_BUCKETS[bucket]
    lines = [f"garment: {spec['_garment']}"]
    present = [f for f in C.FIELDS if spec[f]]
    for f in present:
        extra = ""
        if f == "colour_group" and spec[f] in C.RARE_SURFACE_HINT:
            extra = f"  (you MAY use synonyms like: {C.RARE_SURFACE_HINT[spec[f]]})"
        lines.append(f"{f}: {spec[f]}{extra}")
    absent = [f for f in C.FIELDS if not spec[f]]
    return (
        "Target spec:\n" + "\n".join(lines) + "\n"
        f"Do NOT mention any of these (leave them unspecified): {', '.join(absent) or 'none'}.\n"
        f"Write {n} DIVERSE queries, each {b['min_words']}-{b['max_words']} words, all expressing the "
        f"spec. Vary structure and vocabulary. For each query, 'spans' must have a verbatim substring "
        f"for these fields only: {', '.join(present) or 'none'}. Output ONLY the JSON array of objects."
    )


def _parse_objs(text: str) -> List[Dict]:
    s = text.find("["); e = text.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        arr = json.loads(text[s:e + 1])
    except Exception:
        return []
    out = []
    for o in arr:
        if isinstance(o, dict) and str(o.get("q", "")).strip():
            out.append(o)
    return out


def _call(model: str, spec: Dict, bucket: str, n: int, retries: int = 4) -> List[Dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(spec, bucket, n)},
        ],
        "temperature": 1.0,
        "max_tokens": 2000,
    }
    headers = {"Authorization": f"Bearer {C.API_KEY}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            r = requests.post(f"{C.API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
            if r.status_code == 200:
                return _parse_objs(r.json()["choices"][0]["message"]["content"])
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def _rows_from(spec: Dict, bucket: str, model: str, per: int):
    objs = _call(model, spec, bucket, per)
    out_rows = []
    for o in objs:
        q = str(o["q"]).strip()
        spans = o.get("spans") or {}
        ql = q.lower()
        ok = True
        row = {"query": q, "bucket": bucket, "garment": spec["_garment"], "model": model}
        for f in C.FIELDS:
            surface = str(spans.get(f, "")).strip() if spec[f] else ""
            if spec[f] and (not surface or surface.lower() not in ql):
                ok = False  # span not verbatim in query -> reject
            row[f"{f}_surface"] = surface
            row[f"{f}_canon"] = spec[f] or "none"
        if ok:
            out_rows.append(row)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_specs", type=int, default=12)
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--model", default="gpt-5.4-mini", help="single model, or 'train'/'test' for the pool")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/probe.csv")
    a = ap.parse_args()
    if not C.API_KEY:
        raise RuntimeError("CKEY_API_KEY rỗng — kiểm tra .env")

    models = {"train": C.GEN_MODELS_TRAIN, "test": C.GEN_MODELS_TEST}.get(a.model, [a.model])
    rng = random.Random(a.seed)
    buckets = list(C.LENGTH_BUCKETS)
    tasks = []
    for i in range(a.n_specs):
        spec = _sample_spec(rng)
        bucket = rng.choices(buckets, weights=[C.LENGTH_BUCKETS[b]["ratio"] for b in buckets])[0]
        tasks.append((spec, bucket, models[i % len(models)]))

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_rows_from, s, b, m, a.per) for s, b, m in tasks]
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)} specs, {len(rows)} rows", flush=True)

    out = C.ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\n[gen] {len(rows)} rows from {len(tasks)} specs (models={models}) -> {out}")


if __name__ == "__main__":
    main()
