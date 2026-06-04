import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_CACHE_DIR = BASE_DIR / os.getenv("MODEL_CACHE_DIR", "model_cache")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def setup_environment() -> None:
    os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR))
    os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "0"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

setup_environment()

@dataclass(frozen=True)
class Settings:
    cache_dir: str = str(LOCAL_CACHE_DIR)
    db_path: str = str(BASE_DIR / os.getenv("QDRANT_DB_PATH", "db/qdrant_local_db"))
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "fashion_products")
    graph_file: str = str(BASE_DIR / os.getenv("GRAPH_FILE", "data/processed/final_outfit_graph.csv"))
    meta_file: str = str(BASE_DIR / os.getenv("META_FILE", "data/processed/dataset_qwen_completed.csv"))
    image_dir: str = str(BASE_DIR / os.getenv("IMAGE_DIR", "data/raw/images"))
    frontend_dir: str = str(BASE_DIR / os.getenv("FRONTEND_DIR", "src/frontend"))
    log_dir: str = str(BASE_DIR / os.getenv("LOG_DIR", "log"))

    topk_similar: int = int(os.getenv("TOPK_SIMILAR", "8"))
    topk_graph: int = int(os.getenv("TOPK_GRAPH", "6"))
    topk_variants: int = int(os.getenv("TOPK_VARIANTS", "6"))

    graph_hard_min_weight: int = int(os.getenv("GRAPH_HARD_MIN_WEIGHT", "3"))
    graph_preferred_min_weight: int = int(os.getenv("GRAPH_PREFERRED_MIN_WEIGHT", "4"))
    graph_max_hops: int = int(os.getenv("GRAPH_MAX_HOPS", "3"))
    graph_branch_per_hop: int = int(os.getenv("GRAPH_BRANCH_PER_HOP", "3"))
    graph_max_per_pt: int = int(os.getenv("GRAPH_MAX_PER_PT", "2"))

    use_llm_router: bool = os.getenv("USE_LLM_ROUTER", "1") == "1"
    use_query_rewrite: bool = os.getenv("USE_QUERY_REWRITE", "1") == "1"
    strict_metadata_filters: bool = os.getenv("STRICT_METADATA_FILTERS", "1") == "1"
    max_vision_images: int = int(os.getenv("MAX_VISION_IMAGES", "3"))

    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 6)))
    max_session_context: int = int(os.getenv("MAX_SESSION_CONTEXT", "500"))
    session_history_max: int = int(os.getenv("SESSION_HISTORY_MAX", "12"))

    clean_cuda_cache_each_request: bool = os.getenv("CLEAN_CUDA_CACHE_EACH_REQUEST", "1") == "1"

    # Multi-named-vectors retrieval (Step 4 SOTA refactor)
    vector_name_text: str = os.getenv("VECTOR_NAME_TEXT", "text_emb")
    vector_name_image: str = os.getenv("VECTOR_NAME_IMAGE", "image_emb")
    vector_name_sparse: str = os.getenv("VECTOR_NAME_SPARSE", "sparse_bm25")
    sparse_model_path: str = str(BASE_DIR / os.getenv("SPARSE_MODEL_PATH", "data/processed/sparse_tfidf.json"))
    hnsw_m: int = int(os.getenv("HNSW_M", "32"))
    hnsw_ef_construct: int = int(os.getenv("HNSW_EF_CONSTRUCT", "256"))
    encode_batch_text: int = int(os.getenv("ENCODE_BATCH_TEXT", "64"))
    encode_batch_image: int = int(os.getenv("ENCODE_BATCH_IMAGE", "32"))

    siglip_model_id: str = os.getenv("SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
    qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    qwen_vl_model_id: str = os.getenv("QWEN_VL_MODEL_ID", "")
    qwen_text_model_id: str = os.getenv("QWEN_TEXT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    query_llm_model_id: str = os.getenv("QUERY_LLM_MODEL_ID", "")
    query_llm_device_map: str = os.getenv("QUERY_LLM_DEVICE_MAP", "")
    personalization_dir: str = str(BASE_DIR / os.getenv("PERSONALIZATION_DIR", "data/processed/personalization"))
    intent_classifier_dir: str = str(LOCAL_CACHE_DIR / os.getenv("INTENT_CLASSIFIER_DIR", "intent_classifier_deberta"))
    intent_max_length: int = int(os.getenv("INTENT_MAX_LENGTH", "128"))
    use_intent_classifier: bool = os.getenv("USE_INTENT_CLASSIFIER", "1") == "1"
    use_vl_model: bool = os.getenv("USE_VL_MODEL", "0") == "1"
    use_text_llm_for_nlp: bool = os.getenv("USE_TEXT_LLM_FOR_NLP", "1") == "1"
    llm_local_files_only: bool = os.getenv("LLM_LOCAL_FILES_ONLY", "0") == "1"

settings = Settings()