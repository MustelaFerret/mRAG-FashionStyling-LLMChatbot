import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_CACHE_DIR = BASE_DIR / os.getenv("MODEL_CACHE_DIR", "model_cache")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def setup_environment() -> None:
    os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR))
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

setup_environment()

@dataclass(frozen=True)
class Settings:
    cache_dir: str = str(LOCAL_CACHE_DIR)
    db_path: str = str(BASE_DIR / os.getenv("QDRANT_DB_PATH", "db/qdrant_local_db"))
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "fashion_products")
    graph_file: str = str(BASE_DIR / os.getenv("GRAPH_FILE", "data/processed/final_outfit_graph.csv"))
    # P3alpha transaction edges for cold items (md/refine_5.MD): cold-tier #1 before twins
    aux_graph_file: str = str(BASE_DIR / os.getenv("AUX_GRAPH_FILE", "data/processed/p3a_cold_edges.csv"))
    meta_file: str = str(BASE_DIR / os.getenv("META_FILE", "data/processed/dataset_qwen_completed.csv"))
    image_dir: str = str(BASE_DIR / os.getenv("IMAGE_DIR", "data/raw/images"))
    frontend_dir: str = str(BASE_DIR / os.getenv("FRONTEND_DIR", "src/frontend"))
    log_dir: str = str(BASE_DIR / os.getenv("LOG_DIR", "log"))

    topk_similar: int = int(os.getenv("TOPK_SIMILAR", "8"))
    topk_graph: int = int(os.getenv("TOPK_GRAPH", "6"))
    topk_variants: int = int(os.getenv("TOPK_VARIANTS", "6"))
    # items returned to the UI (a "show more" reveals beyond the first few); the grounded reply is
    # still written from only the top `gen_context_items` so it stays concise.
    ui_item_limit: int = int(os.getenv("UI_ITEM_LIMIT", "10"))
    gen_context_items: int = int(os.getenv("GEN_CONTEXT_ITEMS", "5"))

    graph_hard_min_weight: int = int(os.getenv("GRAPH_HARD_MIN_WEIGHT", "3"))
    graph_preferred_min_weight: int = int(os.getenv("GRAPH_PREFERRED_MIN_WEIGHT", "4"))
    graph_max_hops: int = int(os.getenv("GRAPH_MAX_HOPS", "3"))
    graph_branch_per_hop: int = int(os.getenv("GRAPH_BRANCH_PER_HOP", "3"))
    graph_max_per_pt: int = int(os.getenv("GRAPH_MAX_PER_PT", "2"))
    graph_pair_limit: int = int(os.getenv("GRAPH_PAIR_LIMIT", "8"))
    product_type_match_threshold: float = float(os.getenv("PRODUCT_TYPE_MATCH_THRESHOLD", "0.85"))

    use_llm_router: bool = os.getenv("USE_LLM_ROUTER", "1") == "1"
    use_query_rewrite: bool = os.getenv("USE_QUERY_REWRITE", "1") == "1"
    # in-process numpy/CSR hybrid search over the local qdrant collection (md/refine_3.MD):
    # qdrant embedded scans in pure Python (~4.6s/query with filters); this serves ~30ms
    use_local_hybrid: bool = os.getenv("USE_LOCAL_HYBRID", "1") == "1"
    max_vision_images: int = int(os.getenv("MAX_VISION_IMAGES", "3"))

    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 6)))
    max_session_context: int = int(os.getenv("MAX_SESSION_CONTEXT", "500"))
    session_history_max: int = int(os.getenv("SESSION_HISTORY_MAX", "12"))

    clean_cuda_cache_each_request: bool = os.getenv("CLEAN_CUDA_CACHE_EACH_REQUEST", "1") == "1"

    # Multi-named-vectors retrieval (Step 4 SOTA refactor)
    vector_name_text: str = os.getenv("VECTOR_NAME_TEXT", "text_emb")
    vector_name_image: str = os.getenv("VECTOR_NAME_IMAGE", "image_emb")
    vector_name_sparse: str = os.getenv("VECTOR_NAME_SPARSE", "sparse_bm25")
    sparse_model_path: str = str(BASE_DIR / os.getenv("SPARSE_MODEL_PATH", "data/processed/sparse_bm25.json"))
    hnsw_m: int = int(os.getenv("HNSW_M", "32"))
    hnsw_ef_construct: int = int(os.getenv("HNSW_EF_CONSTRUCT", "256"))
    encode_batch_text: int = int(os.getenv("ENCODE_BATCH_TEXT", "64"))
    encode_batch_image: int = int(os.getenv("ENCODE_BATCH_IMAGE", "32"))

    siglip_model_id: str = os.getenv("SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
    qwen_vl_model_id: str = os.getenv("QWEN_VL_MODEL_ID", "")
    qwen_text_model_id: str = os.getenv("QWEN_TEXT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    query_llm_model_id: str = os.getenv("QUERY_LLM_MODEL_ID", "")
    query_llm_device_map: str = os.getenv("QUERY_LLM_DEVICE_MAP", "")
    personalization_dir: str = str(BASE_DIR / os.getenv("PERSONALIZATION_DIR", "data/processed/personalization"))
    compat_dir: str = str(BASE_DIR / os.getenv("COMPAT_DIR", "data/processed/compat"))
    compat_pairing_fallback: bool = os.getenv("COMPAT_PAIRING_FALLBACK", "1") == "1"
    reranker_model_id: str = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-base")
    # on by default: +14.6pp nDCG@10 on the gold set (md/refine_2.MD)
    use_reranker: bool = os.getenv("USE_RERANKER", "1") == "1"
    # default cpu (~0.5-1.2s / 50-doc pool): the 6GB GPU has no headroom for the reranker
    # next to Qwen+SigLIP+DeBERTa (md/refine_2.MD). Set RERANKER_DEVICE=cuda on a bigger GPU.
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")
    # depth 30: pure-rerank nDCG holds vs 50 (0.9397 vs 0.936) at ~40% less CPU (md/refine_3.MD)
    rerank_candidate_depth: int = int(os.getenv("RERANK_CANDIDATE_DEPTH", "30"))
    slot_extractor_dir: str = str(LOCAL_CACHE_DIR / os.getenv("SLOT_EXTRACTOR_DIR", "slot_extractor_deberta"))
    # on by default: rare-colour/paraphrase recovery, held-out value-F1 0.684 -> 0.894 (+0.75GB VRAM)
    use_slot_extractor: bool = os.getenv("USE_SLOT_EXTRACTOR", "1") == "1"
    intent_classifier_dir: str = str(LOCAL_CACHE_DIR / os.getenv("INTENT_CLASSIFIER_DIR", "intent_classifier_deberta"))
    intent_max_length: int = int(os.getenv("INTENT_MAX_LENGTH", "128"))
    use_intent_classifier: bool = os.getenv("USE_INTENT_CLASSIFIER", "1") == "1"
    use_vl_model: bool = os.getenv("USE_VL_MODEL", "0") == "1"
    use_text_llm_for_nlp: bool = os.getenv("USE_TEXT_LLM_FOR_NLP", "1") == "1"
    llm_local_files_only: bool = os.getenv("LLM_LOCAL_FILES_ONLY", "0") == "1"

settings = Settings()