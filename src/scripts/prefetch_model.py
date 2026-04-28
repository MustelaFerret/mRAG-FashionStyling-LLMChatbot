import os

from huggingface_hub import snapshot_download


HF_HOME = os.getenv("HF_HOME", os.getenv("MODEL_CACHE_DIR", "model_cache"))
os.environ["HF_HOME"] = HF_HOME

SIGLIP_MODEL_ID = os.getenv("SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
QWEN_TEXT_MODEL_ID = os.getenv("QWEN_TEXT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
QWEN_VL_MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")

PREFETCH_SIGLIP = os.getenv("PREFETCH_SIGLIP", "1") == "1"
PREFETCH_QWEN_TEXT = os.getenv("PREFETCH_QWEN_TEXT", "1") == "1"
PREFETCH_QWEN_VL = os.getenv("PREFETCH_QWEN_VL", "0") == "1"


def pull_model(repo_id: str) -> str:
    print(f"[prefetch] downloading {repo_id}")
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        resume_download=True,
    )
    print(f"[prefetch] done {repo_id} -> {local_path}")
    return local_path


def main() -> None:
    print(f"[prefetch] HF_HOME={HF_HOME}")
    pulled = []

    if PREFETCH_SIGLIP:
        pulled.append(SIGLIP_MODEL_ID)
        pull_model(SIGLIP_MODEL_ID)

    if PREFETCH_QWEN_TEXT:
        pulled.append(QWEN_TEXT_MODEL_ID)
        pull_model(QWEN_TEXT_MODEL_ID)

    if PREFETCH_QWEN_VL:
        pulled.append(QWEN_VL_MODEL_ID)
        pull_model(QWEN_VL_MODEL_ID)

    if not pulled:
        print("[prefetch] nothing selected. set PREFETCH_* env vars to 1.")
    else:
        print(f"[prefetch] completed: {', '.join(pulled)}")


if __name__ == "__main__":
    main()