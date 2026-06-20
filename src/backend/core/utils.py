import os
import re
from typing import Iterable, List


def strip_accents(text: str) -> str:
    mapping = str.maketrans({
        "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
        "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
        "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
        "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
        "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
        "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
        "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
        "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
        "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
        "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
        "đ": "d",
        "À": "A", "Á": "A", "Ạ": "A", "Ả": "A", "Ã": "A",
        "Â": "A", "Ầ": "A", "Ấ": "A", "Ậ": "A", "Ẩ": "A", "Ẫ": "A",
        "Ă": "A", "Ằ": "A", "Ắ": "A", "Ặ": "A", "Ẳ": "A", "Ẵ": "A",
        "È": "E", "É": "E", "Ẹ": "E", "Ẻ": "E", "Ẽ": "E",
        "Ê": "E", "Ề": "E", "Ế": "E", "Ệ": "E", "Ể": "E", "Ễ": "E",
        "Ì": "I", "Í": "I", "Ị": "I", "Ỉ": "I", "Ĩ": "I",
        "Ò": "O", "Ó": "O", "Ọ": "O", "Ỏ": "O", "Õ": "O",
        "Ô": "O", "Ồ": "O", "Ố": "O", "Ộ": "O", "Ổ": "O", "Ỗ": "O",
        "Ơ": "O", "Ờ": "O", "Ớ": "O", "Ợ": "O", "Ở": "O", "Ỡ": "O",
        "Ù": "U", "Ú": "U", "Ụ": "U", "Ủ": "U", "Ũ": "U",
        "Ư": "U", "Ừ": "U", "Ứ": "U", "Ự": "U", "Ử": "U", "Ữ": "U",
        "Ỳ": "Y", "Ý": "Y", "Ỵ": "Y", "Ỷ": "Y", "Ỹ": "Y",
        "Đ": "D",
    })
    return text.translate(mapping)


def normalize_text(text: str) -> str:
    return strip_accents((text or "").strip().lower())


def normalize_article_id(article_id: str) -> str:
    if article_id is None:
        return ""
    value = str(article_id).strip()
    if not value:
        return ""
    if value.isdigit():
        return value.zfill(10)
    return value


def extract_article_id_from_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"#?([0-9]{9,10})\b", text)
    if not match:
        return ""
    return normalize_article_id(match.group(1))


def parse_numeric_ids(raw_ids: Iterable[str]) -> List[int]:
    parsed: List[int] = []
    for raw in raw_ids:
        try:
            parsed.append(int(raw))
            continue
        except (TypeError, ValueError):
            pass

        try:
            parsed.append(int(float(raw)))
        except (TypeError, ValueError):
            continue
    return parsed


def get_local_model_path(cache_dir: str, repo_id: str) -> str:
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = os.path.join(cache_dir, repo_folder, "snapshots")
    if not os.path.exists(snapshots_dir):
        return repo_id
    # newest by mtime, not lexicographic: snapshot dirs are commit hashes, so sorted()[-1]
    # can return an older revision when several are cached.
    snaps = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snaps:
        return repo_id
    newest = max(snaps, key=lambda d: os.path.getmtime(os.path.join(snapshots_dir, d)))
    return os.path.join(snapshots_dir, newest)
