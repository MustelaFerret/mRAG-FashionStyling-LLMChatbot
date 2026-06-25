from __future__ import annotations

import json
import os
import re
from threading import Thread
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration, TextIteratorStreamer

from src.backend.core.config import settings
from src.backend.core.utils import get_local_model_path, normalize_text
from src.backend.retrieval.constrained import LMFE_AVAILABLE, build_prefix_allowed_tokens_fn, build_tokenizer_data
from src.backend.services.attribute_gazetteer import AttributeGazetteer


INTENT_SIMILAR = "similar_items"
INTENT_GRAPH = "graph_pairing"
INTENT_VARIANT = "color_variant"
INTENT_CHAT = "chit_chat"
INTENT_COMPOSITE = "composite_intent"

# in a pairing query, text BEFORE one of these connectives describes the TARGET item, AFTER it the
# anchor: "blue trousers TO GO WITH this navy shirt" -> target=blue trousers, anchor=navy shirt.
_PAIR_CONNECTIVES = (
    "to go with", "to match", "to wear with", "to pair with", "to go under", "to go over",
    "that goes with", "that go with", "to complement", "goes with", "go with", "pair with",
    "match with", "with this", "with my", "with the", "with these", "with that", "for this", "for my",
)

CHIT_CHAT_SYSTEM_PROMPT = (
    "You are a fashion assistant. Reply in English. "
    "You must never answer the user's question. "
    "If the user greets or says hello, reply: \"Hello! How can I help you?\". "
    "Otherwise, reply exactly: \"Sorry, I can't help with that request because I'm only a clothing recommendation system.\""
)


class QwenMultimodalService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            model_dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            model_dtype = torch.float32
        self.use_torch_compile = os.getenv("TORCH_COMPILE_LLM", "0") == "1"

        self.processor = None
        self.model = None
        self.using_vl_model = False

        if settings.use_vl_model and settings.qwen_vl_model_id:
            try:
                vl_model_path = get_local_model_path(settings.cache_dir, settings.qwen_vl_model_id)
                self.processor = AutoProcessor.from_pretrained(
                    vl_model_path,
                    local_files_only=settings.llm_local_files_only,
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    vl_model_path,
                    torch_dtype=model_dtype,
                    device_map={"": 0} if self.device.type == "cuda" else "cpu",
                    local_files_only=settings.llm_local_files_only,
                )
                self.model.eval()
                self.using_vl_model = True
            except Exception:
                self.processor = None
                self.model = None
                self.using_vl_model = False

        self.nlp_tokenizer = None
        self.nlp_model = None
        self.using_text_llm_for_nlp = False

        if settings.use_text_llm_for_nlp:
            try:
                text_model_path = get_local_model_path(settings.cache_dir, settings.qwen_text_model_id)
                self.nlp_tokenizer = AutoTokenizer.from_pretrained(
                    text_model_path,
                    local_files_only=settings.llm_local_files_only,
                )
                self.nlp_model = AutoModelForCausalLM.from_pretrained(
                    text_model_path,
                    torch_dtype=model_dtype,
                    device_map={"": 0} if self.device.type == "cuda" else "cpu",
                    local_files_only=settings.llm_local_files_only,
                )
                if self.use_torch_compile:
                    try:
                        self.nlp_model = torch.compile(self.nlp_model)
                    except Exception:
                        pass
                self.nlp_model.eval()
                self.using_text_llm_for_nlp = True
            except Exception:
                pass

        self.query_tokenizer = None
        self.query_model = None
        self.using_query_llm = False

        if settings.query_llm_model_id:
            try:
                query_model_path = get_local_model_path(settings.cache_dir, settings.query_llm_model_id)
                self.query_tokenizer = AutoTokenizer.from_pretrained(
                    query_model_path,
                    local_files_only=settings.llm_local_files_only,
                )
                device_map = settings.query_llm_device_map or ({"": 0} if self.device.type == "cuda" else "cpu")
                self.query_model = AutoModelForCausalLM.from_pretrained(
                    query_model_path,
                    torch_dtype=model_dtype,
                    device_map=device_map,
                    local_files_only=settings.llm_local_files_only,
                )
                if self.use_torch_compile:
                    try:
                        self.query_model = torch.compile(self.query_model)
                    except Exception:
                        pass
                self.query_model.eval()
                self.using_query_llm = True
            except Exception:
                self.query_tokenizer = None
                self.query_model = None
                self.using_query_llm = False

        self.intent_tokenizer = None
        self.intent_model = None
        self.using_intent_classifier = False

        if settings.use_intent_classifier:
            try:
                intent_model_path = settings.intent_classifier_dir
                self.intent_tokenizer = AutoTokenizer.from_pretrained(
                    intent_model_path,
                    local_files_only=settings.llm_local_files_only,
                )
                self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                    intent_model_path,
                    local_files_only=settings.llm_local_files_only,
                    torch_dtype=model_dtype,
                    device_map={"": 0} if self.device.type == "cuda" else "cpu",
                )
                self.intent_model.eval()
                label_map_path = os.path.join(intent_model_path, "label_map.json")
                if os.path.exists(label_map_path):
                    try:
                        with open(label_map_path, "r", encoding="utf-8") as f:
                            label_map = json.load(f)
                        id2label = {int(v): str(k) for k, v in label_map.items()}
                        label2id = {str(k): int(v) for k, v in label_map.items()}
                        self.intent_model.config.id2label = id2label
                        self.intent_model.config.label2id = label2id
                    except Exception:
                        pass
                self.using_intent_classifier = True
            except Exception:
                self.intent_tokenizer = None
                self.intent_model = None
                self.using_intent_classifier = False

        self._enforcer_tokenizer_data = None
        self._filter_schema = None
        self._filter_schema_signature = None
        self.gazetteer = AttributeGazetteer()
        self._slot_extractor = None  # lazy DeBERTa BIO tagger, ensemble fallback (USE_SLOT_EXTRACTOR)

    def _analysis_tokenizer(self):
        if self.using_query_llm and self.query_tokenizer is not None:
            return self.query_tokenizer
        return self.nlp_tokenizer

    def _build_filter_schema(self) -> Dict:
        # The LLM only extracts product_type (free-text: the taxonomy misses everyday words
        # like parka/anorak, and the model generalises these well) + the query rewrite. The
        # closed-vocabulary fields (colour/fit/occasion/season) are NOT asked of the LLM --
        # it dropped and hallucinated them (md/audit_nlu.md); a deterministic gazetteer
        # extracts those instead.
        return {
            "type": "object",
            "properties": {
                "search_query_en": {"type": "string"},
                "must_filters": {
                    "type": "object",
                    "properties": {"product_type": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
            "required": ["search_query_en", "must_filters"],
            "additionalProperties": False,
        }

    def _build_constrained_fn(self, vocab: Dict[str, List[str]] | None):
        if not LMFE_AVAILABLE or not vocab:
            return None
        tokenizer = self._analysis_tokenizer()
        if tokenizer is None:
            return None
        signature = tuple((k, len(vocab.get(k) or [])) for k in sorted(vocab))
        if self._filter_schema is None or self._filter_schema_signature != signature:
            self._filter_schema = self._build_filter_schema()
            self._filter_schema_signature = signature
        try:
            if self._enforcer_tokenizer_data is None:
                self._enforcer_tokenizer_data = build_tokenizer_data(tokenizer)
            return build_prefix_allowed_tokens_fn(self._enforcer_tokenizer_data, self._filter_schema)
        except Exception:
            return None

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for path in image_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    rgb.thumbnail((768, 768))
                    images.append(rgb.copy())
            except Exception:
                continue
        return images

    def _generate_with_text_llm(
        self,
        messages: List[Dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        prefix_allowed_tokens_fn=None,
    ) -> str:
        if self.nlp_model is None or self.nlp_tokenizer is None:
            raise RuntimeError("Text LLM is not available")

        text_messages = self._strip_to_text_messages(messages)
        prompt_text = self.nlp_tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.nlp_tokenizer([prompt_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        if prefix_allowed_tokens_fn is not None:
            generation_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        output_ids = self.nlp_model.generate(**generation_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.nlp_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _generate_with_query_llm(
        self,
        messages: List[Dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        prefix_allowed_tokens_fn=None,
    ) -> str:
        if self.query_model is None or self.query_tokenizer is None:
            raise RuntimeError("Query LLM is not available")

        text_messages = self._strip_to_text_messages(messages)
        prompt_text = self.query_tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.query_tokenizer([prompt_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        if prefix_allowed_tokens_fn is not None:
            generation_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        output_ids = self.query_model.generate(**generation_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.query_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _strip_to_text_messages(messages: List[Dict]) -> List[Dict]:
        cleaned = []
        for msg in messages or []:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        value = str(block.get("text", "")).strip()
                        if value:
                            parts.append(value)
                text = "\n".join(parts)
            else:
                text = ""
            cleaned.append({"role": role, "content": text})
        return cleaned

    def _chat_generate(
        self,
        messages: List[Dict],
        image_paths: List[str] | None = None,
        pil_images: List[Image.Image] | None = None,
        max_new_tokens: int = 180,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        if not self.using_vl_model or self.processor is None or self.model is None:
            return self._generate_with_text_llm(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs: List[Image.Image] = []
        for img in pil_images or []:
            if img is None:
                continue
            rgb = img.convert("RGB")
            rgb.thumbnail((768, 768))
            image_inputs.append(rgb.copy())
        image_inputs.extend(self._load_images(image_paths or []))

        processor_kwargs = {
            "text": [chat_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _chat_generate_text(
        self,
        messages: List[Dict],
        max_new_tokens: int = 120,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        if self.nlp_model is not None and self.nlp_tokenizer is not None:
            return self._generate_with_text_llm(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        return self._chat_generate(
            messages,
            image_paths=None,
            pil_images=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    def _chat_generate_text_stream(
        self,
        messages: List[Dict],
        max_new_tokens: int = 120,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ):
        if self.nlp_model is None or self.nlp_tokenizer is None:
            return

        text_messages = self._strip_to_text_messages(messages)
        prompt_text = self.nlp_tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.nlp_tokenizer([prompt_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.nlp_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "streamer": streamer,
        }

        def _run_generate():
            self.nlp_model.generate(**generation_kwargs)

        thread = Thread(target=_run_generate, daemon=True)
        thread.start()
        for token in streamer:
            if token:
                yield token

    def _chat_generate_stream(
        self,
        messages: List[Dict],
        image_paths: List[str] | None = None,
        pil_images: List[Image.Image] | None = None,
        max_new_tokens: int = 180,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        if not self.using_vl_model or self.processor is None or self.model is None:
            return self._chat_generate_text_stream(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs: List[Image.Image] = []
        for img in pil_images or []:
            if img is None:
                continue
            rgb = img.convert("RGB")
            rgb.thumbnail((768, 768))
            image_inputs.append(rgb.copy())
        image_inputs.extend(self._load_images(image_paths or []))

        processor_kwargs = {
            "text": [chat_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "streamer": streamer,
        }

        def _run_generate():
            self.model.generate(**generation_kwargs)

        thread = Thread(target=_run_generate, daemon=True)
        thread.start()
        for token in streamer:
            if token:
                yield token

    def generate_answer(self, prompt: str, images: List[Image.Image] | None = None) -> str:
        content_blocks: List[Dict] = []
        for _ in images or []:
            content_blocks.append({"type": "image"})
        content_blocks.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_blocks}]
        return self._chat_generate(
            messages,
            pil_images=images or None,
            max_new_tokens=220,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    def generate_answer_stream(self, prompt: str, images: List[Image.Image] | None = None):
        if images:
            content_blocks: List[Dict] = []
            for _ in images or []:
                content_blocks.append({"type": "image"})
            content_blocks.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content_blocks}]
            stream = self._chat_generate_stream(
                messages,
                pil_images=images,
                max_new_tokens=220,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
            )
            if stream is None:
                yield self.generate_answer(prompt, images=images)
                return
            for token in stream:
                yield token
            return

        content_blocks: List[Dict] = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content_blocks}]
        stream = self._chat_generate_text_stream(
            messages,
            max_new_tokens=220,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        if stream is None:
            yield self.generate_answer(prompt, images=None)
            return
        for token in stream:
            yield token

    def generate_chitchat_response(self, user_query: str) -> str:
        messages = [
            {"role": "system", "content": CHIT_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query or ""},
        ]
        return self._chat_generate_text(
            messages,
            max_new_tokens=120,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

    def generate_chitchat_stream(self, user_query: str):
        messages = [
            {"role": "system", "content": CHIT_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query or ""},
        ]
        stream = self._chat_generate_text_stream(
            messages,
            max_new_tokens=120,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        if stream is None:
            yield self.generate_chitchat_response(user_query)
            return
        for token in stream:
            if token:
                yield token

    @staticmethod
    def _extract_json_object(raw_text: str) -> Dict | None:
        # Parse each balanced {...} candidate instead of one greedy first-{ to last-}
        # span: with two objects in the output ("{bad} ... {good}") the greedy span is
        # invalid JSON and extraction silently failed. Last parseable object wins
        # (models emit the final answer last).
        if not raw_text:
            return None
        candidates: List[Dict] = []
        depth, start = 0, -1
        for i, ch in enumerate(raw_text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(raw_text[start:i + 1])
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
        return candidates[-1] if candidates else None

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        lines = []
        for item in history or []:
            role = str(item.get("role", "")).strip().lower() or "user"
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_filters(filters: Dict | None) -> Dict[str, str]:
        if not isinstance(filters, dict):
            return {}
        allowed = {"product_type", "colour_group", "fit", "occasion", "seasonality"}
        normalized: Dict[str, str] = {}
        for key, value in filters.items():
            if key not in allowed:
                continue
            if isinstance(value, (list, tuple)):
                value = next((str(v).strip() for v in value if str(v).strip()), "")
            value_str = str(value).strip()
            if not value_str:
                continue
            normalized[key] = value_str
        return normalized

    @staticmethod
    def _normalize_must_not(filters: Dict | None) -> Dict[str, List[str]]:
        if not isinstance(filters, dict):
            return {}
        allowed = {"product_type", "colour_group", "fit", "occasion", "seasonality"}
        normalized: Dict[str, List[str]] = {}
        for key, value in filters.items():
            if key not in allowed:
                continue
            if isinstance(value, (list, tuple)):
                values = [str(v).strip() for v in value if str(v).strip()]
                if values:
                    normalized[key] = values
                continue
            value_str = str(value).strip()
            if value_str:
                normalized[key] = [value_str]
        return normalized

    def _normalize_intent(self, intent: str | None) -> str:
        value = str(intent or "").strip().lower()
        if value in {INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT, INTENT_CHAT, INTENT_COMPOSITE}:
            return value
        if value in {"chit_chat", "chitchat", "chit-chat", "smalltalk", "greeting"}:
            return INTENT_CHAT
        if value in {"composite_intent", "composite", "composite intent", "multi_intent", "ambiguous"}:
            return INTENT_COMPOSITE
        if value in {"matching", "match", "pairing", "pair"}:
            return INTENT_GRAPH
        if value in {"similar", "lookalike"}:
            return INTENT_SIMILAR
        if value in {"variant", "color_variant", "colour_variant"}:
            return INTENT_VARIANT
        return INTENT_SIMILAR

    @staticmethod
    def _has_anchor_reference(text: str) -> bool:
        if not text:
            return False
        if re.search(r"(#|\b)\d{6,10}\b", text):
            return True
        anchor_markers = [
            "this",
            "that",
            "these",
            "those",
            "my",
            "same",
            "this one",
            "that one",
            "this item",
            "that item",
            "this look",
            "that look",
            "this outfit",
            "that outfit",
            "the item",
            "the look",
            "the outfit",
        ]
        return any(marker in text for marker in anchor_markers)

    @staticmethod
    def _has_strong_pairing_phrase(text: str) -> bool:
        # explicit pairing verb phrases imply an anchor described in the query itself
        # ("what trousers go with a white shirt") even without a this/that marker.
        strong = ["go with", "goes with", "to match", "pair with", "wear with",
                  "style with", "combine with", "matches with"]
        if any(p in text for p in strong):
            return True
        # adverb-inserted forms the substrings above miss: "goes WELL with", "go REALLY WELL with",
        # "pairs NICELY with". Restricted to pairing-specific adverbs so it does not fire on an
        # incidental "go ... with" ("go to the shop with a friend").
        return bool(re.search(
            r"\b(go|goes|going|pair|pairs|wear|wears|match|matches|style|styles|combine|combines)\s+"
            r"(well|really well|nicely|great|perfectly|best)\s+with\b", text))

    @staticmethod
    def _is_color_variant_request(text: str) -> bool:
        if not text:
            return False
        color_terms = ["color", "colour"]
        variant_terms = ["another", "other", "different", "variant", "colorway", "colourway"]
        if not any(term in text for term in color_terms):
            return False
        if not any(term in text for term in variant_terms):
            return False
        return True

    @staticmethod
    def _is_graph_pairing_request(text: str) -> bool:
        if not text:
            return False
        # bare "pair"/"match" excluded: they fire on "a pair of jeans" / "a matching
        # bag is fine" -> false pairing. Real pairing uses the phrase forms below.
        pairing_terms = [
            "pairing",
            "go with",
            "goes with",
            "wear with",
            "style with",
            "combine with",
            "match with",
            "mix with",
            "mixing with",
        ]
        return any(term in text for term in pairing_terms)

    @staticmethod
    def _looks_like_fashion_query(text: str) -> bool:
        value = normalize_text(text)
        if not value:
            return False
        fashion_terms = [
            "outfit",
            "wear",
            "style",
            "match",
            "pair",
            "shirt",
            "t shirt",
            "tshirt",
            "tee",
            "pyjama",
            "pyjamas",
            "pajama",
            "pajamas",
            "sleepwear",
            "nightwear",
            "loungewear",
            "top",
            "hoodie",
            "sweater",
            "jacket",
            "coat",
            "blazer",
            "dress",
            "skirt",
            "pants",
            "trouser",
            "jean",
            "short",
            "legging",
            "shoe",
            "sneaker",
            "boot",
            "sandal",
            "loafer",
            "heel",
            "bag",
            "handbag",
            "belt",
            "hat",
            "cap",
            "scarf",
            "accessory",
            "color",
            "colour",
        ]
        # word-boundary match: substring matching wrongly fired "hat" in "what",
        # "wear" in "weather", routing small talk to a fashion search.
        tokens = set(re.findall(r"[a-z]+", value))
        single = any(t in tokens for t in fashion_terms if " " not in t)
        multi = any(t in value for t in fashion_terms if " " in t)
        return single or multi or QwenMultimodalService._has_anchor_reference(value)

    def _build_intent_rule_debug(self, query: str, intent_hint: str) -> Dict[str, Any]:
        text = normalize_text(query)
        has_colour = bool(self.gazetteer.extract(query).get("colour_group"))
        return {
            "query_normalized": text,
            "classifier_intent": intent_hint,
            "has_anchor_reference": self._has_anchor_reference(text),
            "is_color_variant_request": self._is_color_variant_request(text),
            "is_graph_pairing_request": self._is_graph_pairing_request(text),
            "strong_pairing": self._has_strong_pairing_phrase(text),
            # "(but) in <colour>" against a referenced item = a colour variant request
            "in_colour_cue": has_colour and bool(re.search(r"\bin\b", text)),
            "looks_fashion": self._looks_like_fashion_query(query),
        }

    def _apply_intent_rules(self, query: str, intent_hint: str) -> tuple[str, Dict[str, Any]]:
        debug = self._build_intent_rule_debug(query, intent_hint)
        has_anchor = debug.get("has_anchor_reference")
        is_variant = debug.get("is_color_variant_request") or debug.get("in_colour_cue")
        # the narrow variant cue ("another/different colour", "colorway") counts as a referent
        # signal; the weak "in <colour>" cue does not (it also fires on plain searches/negations
        # like "nothing in black, a white shirt").
        is_variant_phrase = debug.get("is_color_variant_request")
        is_pairing = debug.get("is_graph_pairing_request")
        strong_pairing = debug.get("strong_pairing")

        if intent_hint in {INTENT_CHAT, INTENT_COMPOSITE}:
            if (has_anchor and is_pairing) or strong_pairing:
                debug["rule_applied"] = "override_to_graph_pairing"
                return INTENT_GRAPH, debug
            if has_anchor and is_variant:
                debug["rule_applied"] = "override_to_color_variant"
                return INTENT_VARIANT, debug
            if intent_hint == INTENT_CHAT and self._looks_like_fashion_query(query):
                debug["rule_applied"] = "override_chit_chat_to_similar"
                return INTENT_SIMILAR, debug
            debug["rule_applied"] = "pass_through"
            return intent_hint, debug
        text = debug.get("query_normalized", "")
        if not text:
            debug["rule_applied"] = "empty_query_default_similar"
            return INTENT_SIMILAR, debug

        # The intent classifier scores 97.9% on the labelset (tests/intent_model_eval.py, 2026-06-25),
        # so trust it -- the former "distrust" rules (force_similar_no_explicit_pairing, is_small_talk)
        # only ever removed correct, confident predictions (0 helped / 12 hurt at conf >= 0.93, see
        # md/audit_intent_model_vs_rules.md). The one residual model failure mode is OVER-firing the
        # referent-requiring intents (graph_pairing / color_variant) on a self-contained SEARCH
        # ("something to wear to the gym" -> graph; "nothing in black, a white shirt" -> variant).
        # Those intents are only meaningful with a referent, so keep the model's call when the TEXT
        # carries a referent signal (anchor word / pairing or variant phrase) and fall back to search
        # otherwise. This gates on referent PRESENCE, not on matching a hardcoded phrase, so it no
        # longer demotes valid pairing like "goes well with this" / "build an outfit around this".
        # rag_service additionally re-honors a variant when a real clicked/session anchor or image
        # exists (the text-only layer here cannot see it).
        if intent_hint in {INTENT_GRAPH, INTENT_VARIANT}:
            referent = has_anchor or is_pairing or strong_pairing or is_variant_phrase
            if not referent:
                debug["rule_applied"] = "demote_referentless_to_similar"
                return INTENT_SIMILAR, debug
            debug["rule_applied"] = "trust_classifier"
            return intent_hint, debug

        # The classifier predicts a generic search; upgrade only when the text unambiguously names a
        # more specific intent (catches the model's rare under-prediction toward search).
        if has_anchor and is_variant:
            debug["rule_applied"] = "explicit_color_variant"
            return INTENT_VARIANT, debug
        if (has_anchor and is_pairing) or strong_pairing:
            debug["rule_applied"] = "explicit_graph_pairing"
            return INTENT_GRAPH, debug

        debug["rule_applied"] = "trust_classifier"
        return intent_hint, debug

    def _classify_intent_local(
        self,
        user_query: str,
        history: List[Dict[str, str]],
        anchor_item: Dict | None = None,
        debug: Dict | None = None,
    ) -> Dict:
        if self.intent_model is None or self.intent_tokenizer is None:
            return {
                "intent": INTENT_SIMILAR,
                "confidence": 0.0,
                "raw_label": "",
                "index": -1,
                "source": "fallback_default",
            }

        clean_query = user_query.strip()
        if not clean_query:
            return {
                "intent": INTENT_SIMILAR,
                "confidence": 1.0,
                "raw_label": "",
                "index": -1,
                "source": "empty_query",
            }

        inputs = self.intent_tokenizer(clean_query, return_tensors="pt", truncation=True, max_length=settings.intent_max_length)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.intent_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        idx = int(torch.argmax(probs).item())
        id2label = getattr(self.intent_model.config, "id2label", {}) or {}

        fallback_map = {
            0: INTENT_SIMILAR,
            1: INTENT_GRAPH,
            2: INTENT_VARIANT,
            3: INTENT_COMPOSITE,
            4: INTENT_CHAT,
        }
        raw_label = id2label.get(idx)
        if raw_label is None:
            raw_label = id2label.get(str(idx))
        if raw_label is None:
            raw_label = fallback_map.get(idx, INTENT_SIMILAR)

        label = self._normalize_intent(raw_label)
        confidence = float(probs[idx].item())

        if debug is not None:
            debug["prompt"] = clean_query
            debug["raw"] = json.dumps({"intent": label, "confidence": round(confidence, 4)}, ensure_ascii=False)

        return {
            "intent": label,
            "confidence": confidence,
            "raw_label": str(raw_label),
            "index": idx,
            "source": "intent_classifier",
        }

    def _get_slot_extractor(self, vocab):
        if self._slot_extractor is None and settings.use_slot_extractor:
            try:
                from src.backend.retrieval.slot_extractor import SlotExtractor
                self._slot_extractor = SlotExtractor(valid=vocab or {})
            except Exception:
                self._slot_extractor = False  # load failed -> don't retry
        return self._slot_extractor or None

    def _target_part(self, query: str) -> str:
        """The TARGET side of a pairing query: text before the connective ("blue trousers TO GO WITH
        this navy shirt" -> "blue trousers"). Whole query if there is no connective."""
        q = (query or "").lower()
        cut = len(q)
        for c in _PAIR_CONNECTIVES:
            i = q.find(c)
            if i != -1:
                cut = min(cut, i)
        return q[:cut]

    def _colour_on_target(self, query: str, vocab: Dict[str, List[str]] | None = None) -> bool:
        """True if a colour modifies the pairing TARGET (before the connective)."""
        return bool(self.gazetteer.extract(self._target_part(query), vocab).get("colour_group"))

    def analyze_user_query(self, query: str, vocab: Dict[str, List[str]] | None = None) -> Dict:
        intent_result = self._classify_intent_local(query, [], anchor_item=None)
        intent_hint_raw = self._normalize_intent(intent_result.get("intent"))
        intent_hint, intent_rules = self._apply_intent_rules(query, intent_hint_raw)
        if intent_hint == INTENT_CHAT:
            # Run the deterministic gazetteer even for chit-chat: a terse follow-up ("no, nothing
            # in red", "without a pattern") classifies as chit-chat in isolation yet carries a real
            # constraint, and the orchestration layer needs those filters to decide whether to
            # continue the previous retrieval intent (rag_service refinement-continuation). The LLM
            # rewrite is still skipped -- only the cheap closed-vocab extraction runs.
            gz = self.gazetteer.extract(query, vocab)
            neg = self.gazetteer.extract_negated(query, vocab)
            return {
                "search_query_en": query,
                "intent_hint": INTENT_CHAT,
                "must_filters": gz,
                # positive wins on a contradiction ("no pattern" -> Solid positive, not must_not Solid)
                "must_not_filters": {f: [v] for f, v in neg.items() if gz.get(f) != v},
                "debug": {
                    "intent_classifier": intent_result,
                    "intent_rules": intent_rules,
                    "skipped_extraction": "chit_chat_filters_only",
                },
            }
        prefix_allowed_tokens_fn = self._build_constrained_fn(vocab)
        prompt = (
            "You are a fashion query analyst. Rewrite the user request into English search keywords "
            "and identify the garment type. Return JSON only with this schema:\n"
            "{\n"
            "  \"search_query_en\": string,\n"
            "  \"must_filters\": {\"product_type\": string}\n"
            "}\n"
            "Rules:\n"
            "- search_query_en: English keywords only (keep colour/style words in it).\n"
            "- product_type: ALWAYS set it to the single garment noun the user names (e.g. \"parka\", "
            "\"trousers\", \"dress\", \"shoes\", \"bag\"). This is the most important field -- do not leave it empty when a garment is named.\n"
            "- Only when the request names NO clothing item at all (e.g. \"something for a party\"), set must_filters to {}.\n"
            "- Examples: \"a leather belt\" -> {\"search_query_en\":\"leather belt\",\"must_filters\":{\"product_type\":\"belt\"}}; "
            "\"grey wool socks\" -> {\"search_query_en\":\"grey wool socks\",\"must_filters\":{\"product_type\":\"socks\"}}.\n"
            "- OUTPUT ONLY VALID JSON. No prose, no markdown.\n"
            f"User request: {query}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            if self.using_query_llm:
                raw = self._generate_with_query_llm(
                    messages,
                    max_new_tokens=120,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
            elif self.nlp_model is not None and self.nlp_tokenizer is not None:
                raw = self._generate_with_text_llm(
                    messages,
                    max_new_tokens=120,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
            else:
                raw = self._chat_generate_text(
                    messages,
                    max_new_tokens=120,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                )
        except Exception:
            raw = ""

        obj = self._extract_json_object(raw) if raw else None
        if not obj:
            debug = {
                "intent_classifier": intent_result,
                "intent_rules": intent_rules,
                "llm_raw": raw,
                "llm_parsed": None,
                "llm_parse_ok": False,
                "constrained": prefix_allowed_tokens_fn is not None,
                "using_query_llm": self.using_query_llm,
                "query_llm_model_id": settings.query_llm_model_id or settings.qwen_text_model_id,
            }
            # The LLM JSON failed to parse, but the rule-based intent and the deterministic
            # gazetteer do NOT depend on it -- keep them instead of discarding the request's
            # understanding (previously this returned intent="" and no filters, sending the
            # user down the wrong path with no constraints).
            gz = self.gazetteer.extract(query, vocab)
            neg = self.gazetteer.extract_negated(query, vocab)
            if intent_hint == INTENT_GRAPH:
                if not self._colour_on_target(query, vocab):
                    gz.pop("colour_group", None)
                neg.pop("colour_group", None)
            return {
                "search_query_en": query,
                "intent_hint": intent_hint,
                "must_filters": gz,
                "must_not_filters": {f: [v] for f, v in neg.items()},
                "debug": debug,
            }

        search_query = str(obj.get("search_query_en", "") or "").strip()
        if not search_query:
            search_query = query
        raw_filters = obj.get("must_filters") if isinstance(obj.get("must_filters"), dict) else {}
        raw_must_not = obj.get("must_not_filters") if isinstance(obj.get("must_not_filters"), dict) else {}
        # LLM supplies product_type only; closed-vocab enums come from the deterministic
        # gazetteer (md/audit_nlu.md). Validate enum targets against the live vocab.
        gazetteer_filters = self.gazetteer.extract(query, vocab)
        # product_type is resolved separately so pairing can take it from the TARGET side only
        gz_pt = gazetteer_filters.pop("product_type", "")
        if intent_hint == INTENT_GRAPH:
            # "shoe to go with this dress" -> target type = shoe (before the connective), not the
            # anchor (dress); likewise keep a colour only when it modifies that target.
            gz_pt = self.gazetteer.extract(self._target_part(query), vocab).get("product_type", "")
            if not self._colour_on_target(query, vocab):
                gazetteer_filters.pop("colour_group", None)
        must_filters = {k: v for k, v in self._normalize_filters(raw_filters).items() if k == "product_type"}
        _slot_added = {}
        # the LLM sometimes returns a vague non-garment word for a query that names no item
        # ("something for a gala" -> "gala attire"/"outfit"); drop it (no real product_type).
        pt = normalize_text(must_filters.get("product_type", ""))
        if pt and any(w in pt for w in ("outfit", "garment", "attire", "clothing", "something",
                                        "anything", "clothes")):
            must_filters.pop("product_type", None)
        must_filters.update(gazetteer_filters)
        if gz_pt:
            # deterministic common-garment product_type beats the flaky 1.5B extraction
            # ("boot" -> Boots, "shoe" -> Other shoe); the LLM still handles rarer / OOV types.
            must_filters["product_type"] = gz_pt
        # ENSEMBLE: DeBERTa slot tagger fills enum fields the gazetteer missed (rare colours /
        # paraphrases). Gazetteer wins on conflict; tagger only fills gaps. (md/slot_extractor_plan.md)
        if settings.use_slot_extractor:
            se = self._get_slot_extractor(vocab)
            if se is not None:
                _slot_added = se.fill_missing(query, must_filters)
                if intent_hint == INTENT_GRAPH and not self._colour_on_target(query, vocab):
                    _slot_added.pop("colour_group", None)  # colour belongs to the anchor in pairing
                for f, v in _slot_added.items():
                    must_filters.setdefault(f, v)
        must_not_filters = self._normalize_must_not(raw_must_not)
        # negation -> exclusion: the gazetteer detects negated attributes ("a dress but not red",
        # "jeans that aren't too skinny") that were previously dropped silently; feed them to
        # must_not so retrieval actually excludes the attribute instead of ignoring the negation.
        negated = self.gazetteer.extract_negated(query, vocab)
        if intent_hint == INTENT_GRAPH:
            negated.pop("colour_group", None)  # colour belongs to the anchor in pairing
        for f, v in negated.items():
            if must_filters.get(f) == v:
                continue  # contradictory ("red ... not red"); the positive wins
            bucket = must_not_filters.setdefault(f, [])
            if v not in bucket:
                bucket.append(v)
        debug = {
            "intent_classifier": intent_result,
            "intent_rules": intent_rules,
            "llm_raw": raw,
            "llm_parsed": obj,
            "llm_parse_ok": True,
            "llm_filters_raw": raw_filters,
            "gazetteer_filters": gazetteer_filters,
            "slot_filters": _slot_added,
            "llm_must_not_raw": raw_must_not,
            "constrained": prefix_allowed_tokens_fn is not None,
            "using_query_llm": self.using_query_llm,
            "query_llm_model_id": settings.query_llm_model_id or settings.qwen_text_model_id,
        }

        return {
            "search_query_en": search_query,
            "intent_hint": intent_hint,
            "must_filters": must_filters,
            "must_not_filters": must_not_filters,
            "debug": debug,
        }
