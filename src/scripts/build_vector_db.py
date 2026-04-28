import os
import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import torch.nn.functional as F
import gc

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

os.environ.setdefault("HF_HOME", os.getenv("MODEL_CACHE_DIR", "model_cache"))

DATA_FILE = os.getenv("DATA_FILE", "data/processed/dataset_final_qwen_filled.csv")
IMAGE_DIR = os.getenv("IMAGE_DIR", "data/raw/images")
DB_PATH = os.getenv("QDRANT_DB_PATH", "db/qdrant_local_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fashion_products")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

START_INDEX = int(os.getenv("START_INDEX", "0"))
RESET_DB = os.getenv("RESET_DB", "1") == "1"


print("qdrant init")
client = QdrantClient(path=DB_PATH)

if RESET_DB:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

print("model loaded")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = os.getenv("SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

@torch.no_grad()
def get_joint_embeddings(image_paths, texts):
    images = []
    valid_indices = []
    
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
                valid_indices.append(i)
        except Exception:
            pass
            
    if not images:
        return None, []

    valid_texts = [texts[i] for i in valid_indices]
    
    inputs = processor(text=valid_texts, images=images, padding="max_length", truncation=True, return_tensors="pt").to(device)
    
    image_features = model.get_image_features(pixel_values=inputs.pixel_values)
    text_features = model.get_text_features(input_ids=inputs.input_ids)
    
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    joint_features = (image_features + text_features) / 2.0
    joint_features = F.normalize(joint_features, p=2, dim=-1)
    
    return joint_features.cpu().numpy(), valid_indices

def main():
    df = pd.read_csv(DATA_FILE, dtype={'article_id': str})
    df['article_id'] = df['article_id'].str.zfill(10)
    df = df[df['refined_description'].notna() & (df['refined_description'] != "")]
    total_items = len(df)
    
    print(f"embedding start from index {START_INDEX}")


    for start_idx in tqdm(range(START_INDEX, total_items, BATCH_SIZE)):
        batch_df = df.iloc[start_idx : start_idx + BATCH_SIZE]
        
        image_paths = []
        texts = []
        payloads = []
        
        for _, row in batch_df.iterrows():
            article_id_str = row['article_id']
            folder = article_id_str[:3]
            img_path = os.path.join(IMAGE_DIR, folder, f"{article_id_str}.jpg")
            
            image_paths.append(img_path)
            texts.append(str(row['refined_description']))
            
            payloads.append({
                "article_id": article_id_str,
                "product_type": str(row.get('product_type_name', '')),
                "product_group": str(row.get('product_group_name', '')),
                "garment_group": str(row.get('garment_group_name', '')),
                "colour_group": str(row.get('colour_group_name', '')),
                "index_name": str(row.get('index_name', '')),
                "section_name": str(row.get('section_name', '')),
                "department": str(row.get('department_name', '')),
                "fit": str(row.get('fit', '')),
                "occasion": str(row.get('occasion', '')),
                "seasonality": str(row.get('seasonality', '')),
                "description": str(row.get('refined_description', '')),
                "image_path": img_path
            })
            
        joint_embeddings, valid_indices = get_joint_embeddings(image_paths, texts)
        batch_points = []
        
        if joint_embeddings is not None:
            for i, valid_idx in enumerate(valid_indices):
                point_id = int(payloads[valid_idx]["article_id"])
                
                batch_points.append(
                    PointStruct(
                        id=point_id,
                        vector=joint_embeddings[i].tolist(),
                        payload=payloads[valid_idx]
                    )
                )
                
            client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

        del joint_embeddings, valid_indices, image_paths, texts, payloads, batch_points
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("database build done")
    info = client.get_collection(COLLECTION_NAME)
    print(f"total vectors: {info.points_count}")

if __name__ == '__main__':
    main()