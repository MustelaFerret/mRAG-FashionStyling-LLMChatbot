# Kiểm định tầng intent: DeBERTa trần vs DeBERTa + rules

Cô lập quyết định intent ở tầng LLM (text-only): so `_classify_intent_local` (model trần) với `_apply_intent_rules` (model + hand-rules). Không đụng retrieval, không đụng tầng bơm-ngữ-cảnh ở rag_service (ảnh/anchor/session).

- **Model trần: 47/48 (97.9%)**
- **Model + rules: 47/48 (97.9%)**
- Hiệu ứng rules trên net: **+0.0%** (rules CỨU 0 câu, rules PHÁ 0 câu)
- Bất đồng model≠rules: **1/48 (2.1%)** → tỉ lệ phải gọi LLM-trọng-tài nếu theo phương án đó
- Confidence khi model ĐÚNG: mean 0.92, min 0.54
- Confidence khi model SAI: mean 0.56, max 0.56  (nếu max thấp → có thể gate theo confidence)

## Theo lớp

| Lớp (gold) | n | Model đúng | Model+rules đúng |
|---|---|---|---|
| similar_items | 12 | 12/12 | 12/12 |
| graph_pairing | 12 | 12/12 | 12/12 |
| color_variant | 8 | 7/8 | 7/8 |
| composite_intent | 6 | 6/6 | 6/6 |
| chit_chat | 10 | 10/10 | 10/10 |

## Các câu rules PHÁ (model đúng → rule sai)

(không có)

## Các câu rules CỨU (model sai → rule đúng)

(không có)

## Câu cả hai cùng sai

| Query | gold | model | rule_final | conf |
|---|---|---|---|---|
| the same one but green | color_variant | chit_chat | similar_items | 0.56 |