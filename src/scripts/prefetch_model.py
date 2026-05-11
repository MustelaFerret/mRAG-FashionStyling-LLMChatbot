import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

HF_HOME = os.getenv("HF_HOME", os.getenv("MODEL_CACHE_DIR", "model_cache"))
os.environ["HF_HOME"] = HF_HOME

SIGLIP_MODEL_ID = os.getenv("SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
QWEN_TEXT_MODEL_ID = os.getenv("QWEN_TEXT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
QWEN_OLD_TEXT_MODEL_ID = os.getenv("QWEN_OLD_TEXT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
QWEN_VL_MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")

PREFETCH_SIGLIP = os.getenv("PREFETCH_SIGLIP", "1") == "1"
PREFETCH_QWEN_TEXT = os.getenv("PREFETCH_QWEN_TEXT", "1") == "1"
PREFETCH_QWEN_OLD = os.getenv("PREFETCH_QWEN_OLD", "0") == "1"
PREFETCH_QWEN_VL = os.getenv("PREFETCH_QWEN_VL", "0") == "1"

def pull_model(repo_id: str) -> str:
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=HF_HOME,
            local_files_only=True
        )
        print(f"model cached: {repo_id}")
        return local_path
    except (LocalEntryNotFoundError, FileNotFoundError, ValueError):
        print(f"downloading: {repo_id}")
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=HF_HOME,
            resume_download=True,
        )
        print(f"downloaded: {repo_id}")
        return local_path

def main() -> None:
    print("prefetch start")
    pulled = []

    if PREFETCH_SIGLIP:
        pulled.append(pull_model(SIGLIP_MODEL_ID))
    if PREFETCH_QWEN_TEXT:
        pulled.append(pull_model(QWEN_TEXT_MODEL_ID))
    if PREFETCH_QWEN_OLD:
        pulled.append(pull_model(QWEN_OLD_TEXT_MODEL_ID))
    if PREFETCH_QWEN_VL:
        pulled.append(pull_model(QWEN_VL_MODEL_ID))

    print("prefetch done")

if __name__ == "__main__":
    main()