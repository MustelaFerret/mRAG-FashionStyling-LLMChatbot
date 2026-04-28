import gc
import json
import math
import os
import re

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


os.environ.setdefault("HF_HOME", os.getenv("MODEL_CACHE_DIR", "model_cache"))

INPUT_FILE = os.getenv("INPUT_FILE", "data/processed/dataset_66k_prepared.csv")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/processed/dataset_final_qwen_filled.csv")
IMAGE_DIR = os.getenv("IMAGE_DIR", "data/raw/images")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
LOCAL_FILES_ONLY = os.getenv("LLM_LOCAL_FILES_ONLY", "1") == "1"
MAX_NEW_TOKENS = int(os.getenv("FILL_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("FILL_TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("FILL_TOP_P", "0.9"))

DEFAULT_RESULT = {
    "fit": "Unspecified/Other",
    "occasion": "Unspecified/Other",
    "seasonality": "Unspecified/Other",
    "refined_description": "",
}

SYSTEM_PROMPT = """You are a strictly precise Senior Fashion Data Annotator.
Your task is to analyze the IMAGE first, then use the METADATA as supplementary context. Output ONLY a valid JSON object.

CRITICAL RULES FOR ENUMS:
1. GARMENTS (clothing): You MUST visually infer and choose a specific value for 'fit', 'occasion', and 'seasonality'. DO NOT use "Unspecified/Other".
2. ACCESSORIES/SHOES/UNDERWEAR: You MUST use "Unspecified/Other" for 'fit', 'occasion', and 'seasonality'.

CRITICAL RULES FOR REFINED DESCRIPTION:
1. VISION FIRST: Describe the specific visual details seen in the IMAGE.
2. NO PARAPHRASING: DO NOT simply rewrite the "Original Desc".
3. NO META-TEXT: DO NOT use phrases like "The image shows", "This is a picture of", or mention packaging.
4. SYNTACTIC DIVERSITY: AVOID repetitive sentence starters. DO NOT start every description with "Crafted from", "Made from", "This [item]", or "A [color]". Mix up your sentence structures. Start with the standout visual feature, the silhouette, or the texture.

--- FEW-SHOT EXAMPLES ---

EXAMPLE 1 (Garment):
Input Metadata: Product Type: T-shirt | Group: Jersey Fancy | Color: Dark Blue | Pattern: Placement print | Original Desc: T-shirt in printed cotton jersey with a chest pocket and sewn-in turn-ups on the sleeves.
Output:
{
  "fit": "Regular/Straight",
  "occasion": "Casual/Streetwear",
  "seasonality": "All-Season",
  "refined_description": "A striking white paisley and stripe placement print defines this dark blue T-shirt. The short sleeves feature sewn-in turn-ups, complementing the classic chest pocket on the printed cotton jersey."
}

EXAMPLE 2 (Accessory):
Input Metadata: Product Type: Bag | Group: Accessories | Color: Grey | Pattern: Solid | Original Desc: Small bag in crocodile-patterned imitation leather with a zip at the top and an adjustable strap.
Output:
{
  "fit": "Unspecified/Other",
  "occasion": "Unspecified/Other",
  "seasonality": "Unspecified/Other",
  "refined_description": "Textured crocodile-patterned imitation leather elevates this small grey bumbag. Secured by a top zip closure, the design incorporates an adjustable strap with a metal buckle for versatile styling across the shoulder or waist."
}
"""


def parse_json_output(raw_text: str) -> dict:
    text = re.sub(r"```json\s*", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return dict(DEFAULT_RESULT)

    if not isinstance(data, dict):
        return dict(DEFAULT_RESULT)

    result = dict(DEFAULT_RESULT)
    for key in result:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    return result


class QwenVLDatasetAnnotator:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=model_dtype,
            local_files_only=LOCAL_FILES_ONLY,
        ).to(self.device)
        self.model.eval()

    def _build_context(self, row: pd.Series) -> str:
        return (
            f"Product Type: {row.get('product_type_name', '')}\n"
            f"Group: {row.get('garment_group_name', '')}\n"
            f"Color: {row.get('colour_group_name', '')}\n"
            f"Pattern: {row.get('graphical_appearance_name', '')}\n"
            f"Original Desc: {row.get('detail_desc', '')}"
        )

    @torch.no_grad()
    def _annotate_one(self, image: Image.Image, context: str) -> dict:
        prompt = f"METADATA:\n{context}\n\nTask: Output ONLY the requested JSON format."
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return parse_json_output(raw_text)

    def _prepare_output_file(self, source_df: pd.DataFrame) -> None:
        if os.path.exists(OUTPUT_FILE):
            return

        schema_df = source_df.iloc[0:0].copy()
        for col in DEFAULT_RESULT:
            schema_df[col] = ""
        schema_df.to_csv(OUTPUT_FILE, index=False)

    def _load_processed_ids(self) -> set[str]:
        if not os.path.exists(OUTPUT_FILE):
            return set()
        try:
            df_out = pd.read_csv(OUTPUT_FILE, dtype={"article_id": str})
        except Exception:
            return set()
        if "article_id" not in df_out.columns:
            return set()
        return set(df_out["article_id"].astype(str).str.zfill(10))

    def run(self) -> None:
        print("model init")
        df = pd.read_csv(INPUT_FILE)
        if len(df) == 0:
            print("all processed")
            return

        self._prepare_output_file(df)
        processed_ids = self._load_processed_ids()

        df_target = df.copy()
        df_target["article_id_str"] = df_target["article_id"].astype(str).str.zfill(10)
        df_target = df_target[~df_target["article_id_str"].isin(processed_ids)]

        total_rows_left = len(df_target)
        if total_rows_left == 0:
            print("all processed")
            return

        num_chunks = math.ceil(total_rows_left / CHUNK_SIZE)
        print(f"rows left: {total_rows_left}, chunks: {num_chunks}")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, total_rows_left)
            chunk_df = df_target.iloc[start_idx:end_idx]

            result_rows: list[dict] = []
            for _, row in chunk_df.iterrows():
                article_id_str = row["article_id_str"]
                folder = article_id_str[:3]
                img_path = os.path.join(IMAGE_DIR, folder, f"{article_id_str}.jpg")

                row_data = row.to_dict()
                row_data.pop("article_id_str", None)

                if not os.path.exists(img_path):
                    row_data.update(
                        {
                            "fit": "Unspecified/Other",
                            "occasion": "Unspecified/Other",
                            "seasonality": "Unspecified/Other",
                            "refined_description": "No image available",
                        }
                    )
                    result_rows.append(row_data)
                    continue

                try:
                    with Image.open(img_path) as img:
                        image = img.convert("RGB")
                        image.thumbnail((1024, 1024))
                        parsed = self._annotate_one(image, self._build_context(row))
                except Exception:
                    parsed = dict(DEFAULT_RESULT)

                row_data.update(parsed)
                result_rows.append(row_data)

            if result_rows:
                pd.DataFrame(result_rows).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    annotator = QwenVLDatasetAnnotator()
    annotator.run()


if __name__ == "__main__":
    main()