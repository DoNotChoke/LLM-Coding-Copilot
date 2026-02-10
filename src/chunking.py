from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\nclass ", "\ndef ", "\nasync def ",
        "\n\n", "\n", " ", ""
    ],
    keep_separator=True,
)
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def split_code_file(repo_root: Path, file_path: Path, repo: str, commit: str):
    rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
    text = read_text(file_path)
    if not text.strip():
        return []
    docs = splitter.create_documents(
        [text],
        metadatas=[{
            "repo": repo,
            "commit": commit,
            "file_path": rel,
            "language": file_path.suffix.lstrip(".").lower(),
        }],
    )

    out = []
    for i, d in enumerate(docs):
        chunk_text = d.page_content
        meta = d.metadata

        chunk_hash = sha1_hex(f"{meta['repo']}|{meta['file_path']}|{i}|{chunk_text}")
        pk = chunk_hash

        out.append({
            "pk": pk,
            "text": chunk_text,
            **meta,
            "chunk_index": i,
            "chunk_hash": chunk_hash,
        })
    return out