from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from src.backend.core.config import settings
from src.backend.core.utils import get_local_model_path


INTENT_SIMILAR = "similar_items"
INTENT_GRAPH = "graph_pairing"
INTENT_VARIANT = "color_variant"


class QwenMultimodalService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

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
                self.nlp_model.eval()
                self.using_text_llm_for_nlp = True
            except Exception:
                pass

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

    def _normalize_analysis_output(self, obj: Dict | None, user_query: str) -> Dict:
        if not isinstance(obj, dict):
            obj = {}
        search_query = str(obj.get("search_query", "") or "").strip()
        if not search_query:
            search_query = str(user_query or "").strip()
        must_filters = self._normalize_filters(obj.get("must_filters"))
        must_not_filters = self._normalize_must_not(obj.get("must_not_filters"))
        return {
            "search_query": search_query,
            "must_filters": must_filters,
            "must_not_filters": must_not_filters,
        }

    def _log_analysis(self, session_id: str, prompt: str, raw_text: str, result: Dict) -> None:
        log_dir = settings.log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "llm_router.log")
        timestamp = datetime.utcnow().isoformat()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"timestamp={timestamp}\n")
            f.write(f"session_id={session_id}\n")
            f.write("prompt=\n")
            f.write(prompt.strip() + "\n")
            f.write("raw=\n")
            f.write((raw_text or "").strip() + "\n")
            f.write("json=\n")
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.write("\n")

    def _classify_intent_agent(
        self,
        query: str,
        history: str,
        anchor: str,
        debug: Dict | None = None,
    ) -> str:
        prompt = (
            "Task: Classify user intent for a fashion search system.\n"
            "Options:\n"
            "- 'graph_pairing': User wants an item to WEAR WITH or MATCH the anchor or previous items.\n"
            "- 'color_variant': User wants a DIFFERENT COLOR of the anchor or previous items.\n"
            "- 'similar_items': User is searching for a new item, or no clear pairing/variant intent.\n\n"
            f"[Context]\nHistory: {history}\nAnchor: {anchor}\nQuery: {query}\n\n"
            "Output ONLY the exact string from the options: graph_pairing, color_variant, or similar_items."
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw = ""
        try:
            raw = self._chat_generate_text(
                messages,
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            ).strip()
        except Exception:
            raw = ""

        if debug is not None:
            debug["prompt"] = prompt
            debug["raw"] = raw

        lowered = raw.lower()
        if "graph_pairing" in lowered:
            return INTENT_GRAPH
        if "color_variant" in lowered or "colour_variant" in lowered:
            return INTENT_VARIANT
        if "similar_items" in lowered:
            return INTENT_SIMILAR
        return INTENT_SIMILAR

    def _extract_filters_agent(
        self,
        intent: str,
        query: str,
        history: str,
        anchor: str,
        debug: Dict | None = None,
    ) -> Dict:
        prompt = (
            f"Task: Extract constraints for a fashion database based on the intent '{intent}'.\n"
            "1. 'search_query': A clean search string. Resolve pronouns (it/them) using History.\n"
            "2. 'must_filters': Required attributes (e.g., {'product_type': 'dress'}).\n"
            "3. 'must_not_filters': Prohibited attributes from negative words like 'no', 'avoid', 'hate'.\n\n"
            f"[Context]\nHistory: {history}\nAnchor: {anchor}\nQuery: {query}\n\n"
            "Output ONLY valid JSON:\n"
            "{\"search_query\": \"...\", \"must_filters\": {}, \"must_not_filters\": {}}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw = ""
        try:
            raw = self._chat_generate_text(
                messages,
                max_new_tokens=256,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            obj = self._extract_json_object(raw)
            normalized = self._normalize_analysis_output(obj, query) if obj else {}
        except Exception:
            normalized = {}

        if debug is not None:
            debug["prompt"] = prompt
            debug["raw"] = raw

        return normalized

    def analyze_request(
        self,
        user_query: str,
        history: List[Dict[str, str]],
        anchor_item: Dict | None = None,
        session_id: str = "",
    ) -> Dict:
        anchor_text = ""
        if anchor_item:
            anchor_text = (
                f"#{anchor_item.get('article_id', '')} "
                f"{anchor_item.get('product_type', '')} "
                f"{anchor_item.get('colour_group', '')}"
            ).strip()

        history_text = "None"
        if history:
            history_lines = []
            for msg in history[-4:]:
                role = msg.get("role", "user")
                content = msg.get("text", msg.get("content", ""))
                history_lines.append(f"{role.capitalize()}: {content}")
            history_text = "\n".join(history_lines)

        intent_debug: Dict = {}
        extract_debug: Dict = {}
        intent_raw = self._classify_intent_agent(user_query, history_text, anchor_text, debug=intent_debug)
        intent = self._normalize_intent(intent_raw)
        extraction = self._extract_filters_agent(intent, user_query, history_text, anchor_text, debug=extract_debug)

        result = {
            "intent": intent,
            "search_query": extraction.get("search_query", user_query),
            "must_filters": extraction.get("must_filters", {}),
            "must_not_filters": extraction.get("must_not_filters", {}),
        }

        prompt_log = (
            "### intent_agent_prompt\n"
            f"{intent_debug.get('prompt', '')}\n\n"
            "### extraction_agent_prompt\n"
            f"{extract_debug.get('prompt', '')}"
        )
        raw_log = (
            "intent_raw:\n"
            f"{intent_debug.get('raw', '')}\n\n"
            "extraction_raw:\n"
            f"{extract_debug.get('raw', '')}"
        )
        self._log_analysis(session_id, prompt_log, raw_log, result)
        return result

    def route_intent(self, user_query: str, anchor_item: Dict | None = None) -> str | None:
        anchor_text = ""
        if anchor_item:
            anchor_text = (
                f"#{anchor_item.get('article_id', '')} "
                f"{anchor_item.get('product_type', '')} "
                f"{anchor_item.get('colour_group', '')}"
            ).strip()

        prompt = (
            "Classify the user request into exactly one intent label: "
            "similar_items, graph_pairing, color_variant.\n"
            "Rules:\n"
            "- color_variant: user asks for another color of the same design, or asks if a specific color is available.\n"
            "- graph_pairing: user asks what items match/pair/go with the selected item (outfit building).\n"
            "- similar_items: user asks for look-alike/similar items.\n"
            "Disambiguation:\n"
            "- If user asks for matching/pairing with current item, choose graph_pairing even if color words appear.\n"
            "- If user only asks about color availability/change, choose color_variant.\n"
            "- If user asks for lookalikes, choose similar_items.\n"
            "Examples:\n"
            "1) 'I want to find things to mix with this' -> graph_pairing\n"
            "2) 'What can I wear with this hoodie?' -> graph_pairing\n"
            "3) 'Show similar hoodies like this one' -> similar_items\n"
            "4) 'Do you have this in black color?' -> color_variant\n"
            "Return only JSON with key intent, example: {\"intent\":\"similar_items\"}. "
            f"Anchor item: {anchor_text or 'unknown'}\n"
            f"User request: {user_query}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            raw = self._chat_generate_text(messages, max_new_tokens=56, do_sample=False, temperature=0.0, top_p=1.0)
        except Exception:
            return None

        obj = self._extract_json_object(raw)
        if not obj:
            return None

        intent = obj.get("intent")
        if intent in {INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT}:
            return intent
        return None

    def detect_pairing_request(self, user_query: str, anchor_item: Dict | None = None) -> bool | None:
        anchor_text = ""
        if anchor_item:
            anchor_text = (
                f"#{anchor_item.get('article_id', '')} "
                f"{anchor_item.get('product_type', '')} "
                f"{anchor_item.get('colour_group', '')}"
            ).strip()

        prompt = (
            "Determine if the user is asking for complementary items to wear/pair with the current anchor item.\n"
            "Return only JSON: {\"pairing_request\": true} or {\"pairing_request\": false}.\n"
            "Guidelines:\n"
            "- true: asks to mix/match/pair/wear with this/that item, even if no target category (pants/shoes) is explicitly stated.\n"
            "- false: asks for similar items or asks for color variants of the same item.\n"
            "Examples:\n"
            "1) 'I want to find things to mix with this' -> true\n"
            "2) 'Find pants that go with this hoodie' -> true\n"
            "3) 'Show similar hoodies like this' -> false\n"
            "4) 'Do you have this in black?' -> false\n"
            f"Anchor item: {anchor_text or 'unknown'}\n"
            f"User request: {user_query}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            raw = self._chat_generate_text(messages, max_new_tokens=40, do_sample=False, temperature=0.0, top_p=1.0)
        except Exception:
            return None

        obj = self._extract_json_object(raw)
        if not obj:
            return None

        value = obj.get("pairing_request")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return None

    def extract_query_constraints(self, user_query: str, anchor_item: Dict | None = None) -> Dict | None:
        anchor_text = ""
        if anchor_item:
            anchor_text = (
                f"#{anchor_item.get('article_id', '')} "
                f"{anchor_item.get('product_type', '')} "
                f"{anchor_item.get('colour_group', '')}"
            ).strip()

        prompt = (
            "Extract structured retrieval constraints from the user query for fashion search.\n"
            "Return JSON only with this schema:\n"
            "{\n"
            "  \"search_query\": string,\n"
            "  \"intent_hint\": \"similar_items\" | \"graph_pairing\" | \"color_variant\" | \"\",\n"
            "  \"include_filters\": {\"product_type\":\"\",\"colour_group\":\"\",\"fit\":\"\",\"occasion\":\"\",\"seasonality\":\"\"},\n"
            "  \"exclude_filters\": {\"colour_group\": [string]},\n"
            "  \"notes\": string\n"
            "}\n"
            "Rules:\n"
            "- Keep include_filters empty if unsure.\n"
            "- Put negations like 'not black/white' into exclude_filters.colour_group.\n"
            "- Preserve attribute binding in search_query (e.g., black jacket + white trousers).\n"
            "- For queries asking what to mix/match with current item, set intent_hint=graph_pairing.\n"
            f"Anchor item: {anchor_text or 'unknown'}\n"
            f"User request: {user_query}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            raw = self._chat_generate_text(messages, max_new_tokens=180, do_sample=False, temperature=0.0, top_p=1.0)
        except Exception:
            return None

        obj = self._extract_json_object(raw)
        if not obj:
            return None

        include_filters = obj.get("include_filters") if isinstance(obj.get("include_filters"), dict) else {}
        exclude_filters = obj.get("exclude_filters") if isinstance(obj.get("exclude_filters"), dict) else {}
        intent_hint = str(obj.get("intent_hint", "") or "").strip()
        search_query = str(obj.get("search_query", "") or "").strip()

        normalized_exclude: Dict[str, List[str]] = {}
        colour_excludes = exclude_filters.get("colour_group")
        if isinstance(colour_excludes, list):
            normalized_exclude["colour_group"] = [str(v).strip() for v in colour_excludes if str(v).strip()]
        elif isinstance(colour_excludes, str) and colour_excludes.strip():
            normalized_exclude["colour_group"] = [colour_excludes.strip()]

        return {
            "search_query": search_query,
            "intent_hint": intent_hint,
            "include_filters": {
                "product_type": str(include_filters.get("product_type", "") or "").strip(),
                "colour_group": str(include_filters.get("colour_group", "") or "").strip(),
                "fit": str(include_filters.get("fit", "") or "").strip(),
                "occasion": str(include_filters.get("occasion", "") or "").strip(),
                "seasonality": str(include_filters.get("seasonality", "") or "").strip(),
            },
            "exclude_filters": normalized_exclude,
            "notes": str(obj.get("notes", "") or "").strip(),
            "model": settings.qwen_text_model_id if self.using_text_llm_for_nlp else settings.qwen_vl_model_id,
        }

    def rewrite_search_query(
        self,
        user_query: str,
        anchor_item: Dict,
        intent_label: str,
        image: Image.Image | None = None,
    ) -> str | None:
        anchor_text = (
            f"#{anchor_item.get('article_id', '')} "
            f"{anchor_item.get('product_type', '')} "
            f"{anchor_item.get('colour_group', '')} "
            f"{anchor_item.get('description', '')}"
        ).strip()

        prompt = (
            "Rewrite the user request into a short retrieval query for fashion item search. "
            "Keep only concrete searchable attributes and item types. "
            "Remove filler words and commands. "
            "Return JSON only with key search_query. "
            "Example: {\"search_query\":\"blue denim jeans black khaki pants smart casual\"}.\n"
            f"Intent: {intent_label}\n"
            f"Anchor item: {anchor_text or 'unknown'}\n"
            f"User request: {user_query or 'match outfit from image'}"
        )

        content_blocks: List[Dict] = []
        if image is not None:
            content_blocks.append({"type": "image"})
        content_blocks.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_blocks}]

        try:
            raw = self._chat_generate(
                messages,
                pil_images=[image] if image is not None else None,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception:
            return None

        if not raw:
            return None

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                query = str(payload.get("search_query", "")).strip()
                return query[:160] if query else None
            except json.JSONDecodeError:
                pass

        first_line = raw.splitlines()[0].strip().strip('"').strip("'")
        if not first_line:
            return None
        return first_line[:160]

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
