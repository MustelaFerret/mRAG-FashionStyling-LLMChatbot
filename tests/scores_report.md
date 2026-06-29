# Điểm thi NLU end-to-end (multi-turn)

Mỗi case là một **hội thoại nhiều lượt** chung session, chạy qua toàn bộ pipeline production (`FashionRAGService.prepare_chat`): intent classifier → intent rules → gazetteer → Qwen rewrite → filtered hybrid retrieval → rerank. Chấm **từng lượt**; một hội thoại đạt chỉ khi **mọi lượt** của nó đạt.

- Đề thi: `tests/exam_cases.json` · Đáp án: `tests/answer_key.json`
- **Hội thoại: 22/22 (100.0%)** · **Lượt: 34/34 (100.0%)**

## Tổng hợp theo độ khó

| Tier | Hội thoại đạt | Lượt đạt |
|---|---|---|
| easy | 6/6 | 6/6 |
| medium | 5/5 | 5/5 |
| hard | 11/11 | 23/23 |

## Chi tiết từng hội thoại

### Tier: easy

#### [PASS] E1 — type + colour (dress, black)
- **[PASS] E1** `a black dress`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Dress', 'colour_group': 'Black'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Dress', 'Dress', 'Dress', 'Dress', 'Dress', 'Dress']

#### [PASS] E2 — synonym type (jeans -> Trousers)
- **[PASS] E2** `blue jeans`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Trousers', 'colour_group': 'Blue'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers']

#### [PASS] E3 — type + colour (sweater, grey)
- **[PASS] E3** `a grey sweater`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Sweater', 'colour_group': 'Grey'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater']

#### [PASS] E4 — accessory type (bag)
- **[PASS] E4** `a leather bag`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Bag'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag']

#### [PASS] E5 — occasion-only (loungewear, no type)
- **[PASS] E5** `comfy loungewear set for sleeping`
  - intent=`similar_items` (cls=`similar_items`) · must={'occasion': 'Lounge/Sleep/Nightwear'} · must_not={}
  - path=['filter+occasion', 'cross_encoder_rerank'] · 10 kết quả · types=['Pyjama set', 'Pyjama set', 'Pyjama set', 'Pyjama set', 'Pyjama set', 'Pyjama set']

#### [PASS] E6 — occasion-only (gym, no type)
- **[PASS] E6** `something to wear to the gym`
  - intent=`similar_items` (cls=`graph_pairing`) · must={'occasion': 'Sport/Active/Workout'} · must_not={}
  - path=['filter+occasion', 'cross_encoder_rerank'] · 10 kết quả · types=['T-shirt', 'T-shirt', 'T-shirt', 'Shorts', 'T-shirt', 'T-shirt']

### Tier: medium

#### [PASS] M1 — type + occasion (dress, wedding)
- **[PASS] M1** `a dress for a wedding`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Dress', 'occasion': 'Party/Evening/Wedding'} · must_not={}
  - path=['filter+occasion', 'cross_encoder_rerank'] · 10 kết quả · types=['Dress', 'Dress', 'Dress', 'Dress', 'Dress', 'Dress']

#### [PASS] M2 — type + gender (shirt, men)
- **[PASS] M2** `a shirt for men`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Shirt'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank', 'gender_filter:Men'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']

#### [PASS] M3 — type + colour + fit (trousers, black, slim)
- **[PASS] M3** `slim fit black trousers`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Trousers', 'colour_group': 'Black', 'fit': 'Slim/Tailored'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers']

#### [PASS] M4 — negation colour (sweater, not black)
- **[PASS] M4** `a sweater but not black`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Sweater'} · must_not={'colour_group': ['Black']}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater']

#### [PASS] M5 — OOV type + season (parka -> Jacket, winter)
- **[PASS] M5** `a parka for cold weather`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Jacket', 'seasonality': 'Autumn/Winter'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Jacket', 'Jacket', 'Jacket', 'Jacket', 'Jacket', 'Jacket']

### Tier: hard

#### [PASS] C1 — refine colour, then switch type
- **[PASS] C1.1** `I want a white shirt`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Shirt', 'colour_group': 'White'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']
- **[PASS] C1.2** `make it blue`
  - intent=`similar_items` (cls=`color_variant`) · must={'colour_group': 'Blue', 'product_type': 'Shirt'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']
- **[PASS] C1.3** `now show me black trousers instead`
  - intent=`similar_items` (cls=`graph_pairing`) · must={'product_type': 'Trousers', 'colour_group': 'Black'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers']

#### [PASS] C2 — search, then pair with the item just found
- **[PASS] C2.1** `find me a white shirt`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Shirt', 'colour_group': 'White'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']
- **[PASS] C2.2** `what goes well with this`
  - intent=`graph_pairing` (cls=`graph_pairing`, anchor=0681900001) · must={} · must_not={}
  - path=['intent_graph_pairing', 'anchor_from_selection', 'graph_neighbors', 'compat_pairing_fallback', 'pairing_query_rerank'] · 8 kết quả · types=['Hat/beanie', 'Other accessories', 'Scarf', 'Earring', 'Scarf', 'Hat/beanie']

#### [PASS] C3 — search, then colour variant of the item just found
- **[PASS] C3.1** `show me a black dress`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Dress', 'colour_group': 'Black'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Dress', 'Dress', 'Dress', 'Dress', 'Dress', 'Dress']
- **[PASS] C3.2** `do you have this in red`
  - intent=`color_variant` (cls=`color_variant`, anchor=0842290001) · must={'product_type': 'Shirt', 'colour_group': 'Red'} · must_not={}
  - path=['color_variant_image_knn'] · 10 kết quả · types=['Dress', 'Dress', 'Dress', 'Dress', 'Dress', 'Dress']

#### [PASS] C4 — negation refinement across turns
- **[PASS] C4.1** `a sweater`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Sweater'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater']
- **[PASS] C4.2** `but not black`
  - intent=`similar_items` (cls=`color_variant`) · must={'product_type': 'Sweater'} · must_not={'colour_group': ['Black']}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater', 'Sweater']

#### [PASS] C5 — chit-chat, then recover to a real search
- **[PASS] C5.1** `hi there, what can you do?`
  - intent=`chit_chat` (cls=`chit_chat`, direct=`chit_chat`) · must={} · must_not={}
  - path=['intent_chit_chat'] · 0 kết quả · types=[]
- **[PASS] C5.2** `ok, show me a red dress`
  - intent=`similar_items` (cls=`chit_chat`) · must={'product_type': 'Dress', 'colour_group': 'Red'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Dress', 'Dress', 'Dress', 'Dress', 'Dress', 'Dress']

#### [PASS] C6 — multi-step accumulating refinement
- **[PASS] C6.1** `I want a white shirt`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Shirt', 'colour_group': 'White'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']
- **[PASS] C6.2** `make it slim fit`
  - intent=`similar_items` (cls=`chit_chat`) · must={'fit': 'Slim/Tailored', 'product_type': 'Shirt'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']
- **[PASS] C6.3** `actually in black`
  - intent=`similar_items` (cls=`color_variant`) · must={'colour_group': 'Black', 'product_type': 'Shirt'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']

#### [PASS] C7 — search trousers, then build an outfit around it
- **[PASS] C7.1** `find black trousers`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Trousers', 'colour_group': 'Black'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers']
- **[PASS] C7.2** `build an outfit around this`
  - intent=`graph_pairing` (cls=`graph_pairing`, anchor=0612075008) · must={} · must_not={}
  - path=['intent_graph_pairing', 'anchor_from_selection', 'graph_neighbors', 'cold_twin_compat_rrf', 'pairing_query_rerank'] · 8 kết quả · types=['Cardigan', 'T-shirt', 'Vest top', 'Cardigan', 'Vest top', 'Sweater']

#### [PASS] C8 — implicit pronoun pairing via phrase (chat-rescue)
- **[PASS] C8.1** `find black trousers`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Trousers', 'colour_group': 'Black'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers']
- **[PASS] C8.2** `what goes well with it`
  - intent=`graph_pairing` (cls=`chit_chat`, anchor=0612075008) · must={} · must_not={}
  - path=['intent_graph_pairing', 'anchor_from_selection', 'graph_neighbors', 'cold_twin_compat_rrf', 'pairing_query_rerank'] · 8 kết quả · types=['Hat/beanie', 'Hat/beanie', 'Other accessories', 'Scarf', 'Hat/beanie', 'Other accessories']

#### [PASS] C9 — phraseless pairing (graph re-honor via real anchor)
- **[PASS] C9.1** `find black trousers`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Trousers', 'colour_group': 'Black'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers', 'Trousers']
- **[PASS] C9.2** `what matches it`
  - intent=`graph_pairing` (cls=`graph_pairing`, anchor=0612075008) · must={} · must_not={}
  - path=['intent_graph_pairing', 'anchor_from_selection', 'graph_neighbors', 'cold_twin_compat_rrf', 'pairing_query_rerank'] · 8 kết quả · types=['Hat/beanie', 'Hat/beanie', 'Scarf', 'Other accessories', 'Other accessories', 'Hat/beanie']

#### [PASS] C10 — variant without a target colour (not swept into refine)
- **[PASS] C10.1** `show me a black hoodie`
  - intent=`similar_items` (cls=`similar_items`) · must={'colour_group': 'Black', 'product_type': 'Hoodie'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank'] · 10 kết quả · types=['Hoodie', 'Hoodie', 'Hoodie', 'Hoodie', 'Hoodie', 'Hoodie']
- **[PASS] C10.2** `any different color of this one?`
  - intent=`color_variant` (cls=`color_variant`, anchor=0715624054) · must={'product_type': 'Dress'} · must_not={}
  - path=['color_variant_other_colours'] · 10 kết quả · types=['Hoodie', 'Hoodie', 'Hoodie', 'Hoodie', 'Hoodie', 'Hoodie']

#### [PASS] C11 — no <type> with the term -> cross-category suggestion
- **[PASS] C11** `I want a shirt that has the word NASA printed on it`
  - intent=`similar_items` (cls=`similar_items`) · must={'product_type': 'Shirt'} · must_not={}
  - path=['hard_filters', 'cross_encoder_rerank', 'cross_category_suggestion'] · 10 kết quả · types=['Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt', 'Shirt']
