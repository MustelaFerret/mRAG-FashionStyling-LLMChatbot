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


INTENT_SIMILAR = "similar_items"
INTENT_GRAPH = "graph_pairing"
INTENT_VARIANT = "color_variant"


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
                self.using_intent_classifier = True
            except Exception:
                self.intent_tokenizer = None
                self.intent_model = None
                self.using_intent_classifier = False

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

    @staticmethod
    def _extract_json_object(raw_text: str) -> Dict | None:
        if not raw_text:
            return None

        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return None

        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

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
        if value in {INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT}:
            return value
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
        pairing_terms = [
            "pair",
            "pairing",
            "match",
            "matching",
            "go with",
            "goes with",
            "wear with",
            "style with",
            "combine with",
            "match with",
        ]
        return any(term in text for term in pairing_terms)

    def _build_intent_rule_debug(self, query: str, intent_hint: str) -> Dict[str, Any]:
        text = normalize_text(query)
        has_anchor = self._has_anchor_reference(text)
        is_color_variant = self._is_color_variant_request(text)
        is_graph_pairing = self._is_graph_pairing_request(text)
        return {
            "query_normalized": text,
            "classifier_intent": intent_hint,
            "has_anchor_reference": has_anchor,
            "is_color_variant_request": is_color_variant,
            "is_graph_pairing_request": is_graph_pairing,
        }

    def _apply_intent_rules(self, query: str, intent_hint: str) -> tuple[str, Dict[str, Any]]:
        debug = self._build_intent_rule_debug(query, intent_hint)
        text = debug.get("query_normalized", "")
        if not text:
            debug["rule_applied"] = "empty_query_default_similar"
            return INTENT_SIMILAR, debug

        if debug.get("has_anchor_reference") and debug.get("is_color_variant_request"):
            debug["rule_applied"] = "explicit_color_variant"
            return INTENT_VARIANT, debug
        if debug.get("has_anchor_reference") and debug.get("is_graph_pairing_request"):
            debug["rule_applied"] = "explicit_graph_pairing"
            return INTENT_GRAPH, debug
        if intent_hint in {INTENT_GRAPH, INTENT_VARIANT}:
            debug["rule_applied"] = "force_similar_no_explicit_pairing"
            return INTENT_SIMILAR, debug

        debug["rule_applied"] = "default_similar"
        return INTENT_SIMILAR, debug

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

        # CHỈ đưa câu query thuần túy vào mô hình, giới hạn đúng 32 tokens như lúc train
        inputs = self.intent_tokenizer(clean_query, return_tensors="pt", truncation=True, max_length=32)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.intent_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        idx = int(torch.argmax(probs).item())
        id2label = getattr(self.intent_model.config, "id2label", {}) or {}
        
        # Nếu model chưa lưu id2label, hardcode fallback dự phòng
        fallback_map = {0: INTENT_SIMILAR, 1: INTENT_GRAPH, 2: INTENT_VARIANT}
        raw_label = id2label.get(idx) or fallback_map.get(idx, INTENT_SIMILAR)
        
        label = self._normalize_intent(raw_label)
        confidence = float(probs[idx].item())

        if debug is not None:
            debug["prompt"] = clean_query  # Chỉ log query sạch
            debug["raw"] = json.dumps({"intent": label, "confidence": round(confidence, 4)}, ensure_ascii=False)

        return {
            "intent": label,
            "confidence": confidence,
            "raw_label": str(raw_label),
            "index": idx,
            "source": "roberta",
        }

    def analyze_user_query(self, query: str) -> Dict:
        intent_result = self._classify_intent_local(query, [], anchor_item=None)
        intent_hint_raw = self._normalize_intent(intent_result.get("intent"))
        intent_hint, intent_rules = self._apply_intent_rules(query, intent_hint_raw)
        prompt = (
            "You are a fashion query analyst. Rewrite the user request into English keywords suitable for search. "
            "Extract filters in one pass. Return JSON only with this schema:\n"
            "{\n"
            "  \"search_query_en\": string,\n"
            "  \"must_filters\": {\"product_type\":\"\",\"colour_group\":\"\",\"fit\":\"\",\"occasion\":\"\",\"seasonality\":\"\"},\n"
            "  \"must_not_filters\": {\"product_type\":[\"\"],\"colour_group\":[\"\"],\"fit\":[\"\"],\"occasion\":[\"\"],\"seasonality\":[\"\"]}\n"
            "}\n"
            "Rules:\n"
            "- search_query_en must be English keywords only.\n"
            "- Keep must_filters empty if unsure.\n"
            "- Put negations into must_not_filters.\n"
            "- intent_hint is handled by a separate classifier. Do not invent it here.\n"
            "- If the user query contains a clothing item (e.g., parka, jacket, hat, dress), YOU MUST extract it into the \"product_type\" field. DO NOT leave it empty.\n"
            "- OUTPUT ONLY VALID JSON. DO NOT output any conversational text, markdown formatting, or explanations.\n"
            "RULES FOR \"intent_hint\":\n"
            "- MUST output \"similar_items\" if the user is looking for a specific piece of clothing based on description, style, or need (e.g., \"I want to find...\", \"Looking for a jacket...\").\n"
            "- MUST output \"graph_pairing\" ONLY if the user explicitly asks to find complementary items to pair with a given/referenced item (e.g., \"What pants go with this jacket?\", \"Find shoes matching this look\").\n"
            "- MUST output \"color_variant\" ONLY if the user asks for the exact same item in a different color.\n"
            f"User request: {query}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            if self.using_query_llm:
                raw = self._generate_with_query_llm(
                    messages,
                    max_new_tokens=100,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                )
            else:
                raw = self._chat_generate_text(
                    messages,
                    max_new_tokens=100,
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
                "using_query_llm": self.using_query_llm,
                "query_llm_model_id": settings.query_llm_model_id or settings.qwen_text_model_id,
            }
            return {
                "search_query_en": query,
                "intent_hint": "",
                "must_filters": {},
                "must_not_filters": {},
                "debug": debug,
            }

        search_query = str(obj.get("search_query_en", "") or "").strip()
        if not search_query:
            search_query = query
        raw_filters = obj.get("must_filters") if isinstance(obj.get("must_filters"), dict) else {}
        raw_must_not = obj.get("must_not_filters") if isinstance(obj.get("must_not_filters"), dict) else {}
        must_filters = self._normalize_filters(raw_filters)
        must_not_filters = self._normalize_must_not(raw_must_not)
        debug = {
            "intent_classifier": intent_result,
            "intent_rules": intent_rules,
            "llm_raw": raw,
            "llm_parsed": obj,
            "llm_parse_ok": True,
            "llm_filters_raw": raw_filters,
            "llm_must_not_raw": raw_must_not,
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

    def summarize(
        self,
        user_query: str,
        intent_label: str,
        anchor_item: Dict,
        result_items: List[Dict],
        image_paths: List[str],
        extra_images: List[Image.Image] | None = None,
    ) -> str:
        lines = []
        for idx, item in enumerate(result_items, 1):
            lines.append(
                f"{idx}. #{item['article_id']} | {item['product_type']} | {item['colour_group']} | fit={item['fit']} | occasion={item['occasion']}"
            )
        items_text = "\n".join(lines)

        prompt = (
            "You are a Styling Assistant for a RAG system. "
            "You must summarize and comment only using the retrieved item list, and you must not suggest items outside the list. "
            "Use both visual and text context, and answer in concise English (2-3 sentences).\n"
            f"Intent: {intent_label}\n"
            f"User query: {user_query or 'image search'}\n"
            f"Anchor: #{anchor_item['article_id']} {anchor_item['product_type']} {anchor_item['colour_group']}\n"
            f"Retrieved items:\n{items_text}"
        )

        vision_images: List[Image.Image] = []
        for img in extra_images or []:
            if img is None:
                continue
            rgb = img.convert("RGB")
            rgb.thumbnail((768, 768))
            vision_images.append(rgb.copy())
        if len(vision_images) < settings.max_vision_images:
            remaining = settings.max_vision_images - len(vision_images)
            vision_images.extend(self._load_images(image_paths[:remaining]))

        content_blocks: List[Dict] = [{"type": "image"} for _ in vision_images]
        content_blocks.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_blocks}]

        try:
            answer = self._chat_generate(
                messages,
                pil_images=vision_images,
                max_new_tokens=180,
                temperature=0.25,
                top_p=0.9,
                do_sample=True,
            )
            if answer:
                return answer
        except Exception:
            pass

        if intent_label == INTENT_GRAPH:
            return "I prioritized items that are most likely to pair with the focus item using the co-buy graph, then filtered them by your request intent."
        if intent_label == INTENT_VARIANT:
            return "I retrieved color and close-style variants of the same design so you can compare options more precisely."
        return "I retrieved similar style items using vector search, with metadata filters applied when available."
