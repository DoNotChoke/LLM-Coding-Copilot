import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from pymilvus import connections

from chunking import split_code_file
from embedding import Embedder
from milvus import ensure_collection, delete_file_chunks, upsert_chunks

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".github", "node_modules", "dist", "build", "__pycache__", ".venv", "venv",
    ".idea", ".vscode", ".pytest_cache", "target", "out",
}

DEFAULT_INCLUDE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".cs",
    ".md", ".yaml", ".yml", ".json", ".toml",
}


def _parse_include_dirs(repo_root: Path, include_dirs_csv: str) -> List[Path]:
    dirs = [d.strip().strip("/\\") for d in (include_dirs_csv or "").split(",") if d.strip()]
    if not dirs:
        dirs = ["src"]  # default
    out: List[Path] = []
    for d in dirs:
        p = (repo_root / d).resolve()
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def _is_under_any(path: Path, roots: List[Path]) -> bool:
    rp = path.resolve()
    for r in roots:
        try:
            rp.relative_to(r.resolve())
            return True
        except ValueError:
            continue
    return False


def iter_repo_files(repo_root: Path, include_roots: List[Path]) -> List[Path]:
    out: List[Path] = []
    for root in include_roots:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            parts = set(p.parts)
            if any(d in parts for d in DEFAULT_EXCLUDE_DIRS):
                continue
            if p.suffix.lower() not in DEFAULT_INCLUDE_EXTS:
                continue
            out.append(p)
    return out


def parse_changed_files_env(repo_root: Path, include_roots: List[Path]) -> Optional[List[Path]]:
    raw = os.getenv("CHANGED_FILES", "").strip()
    if not raw:
        return None
    rels = [line.strip() for line in raw.splitlines() if line.strip()]
    files: List[Path] = []
    for r in rels:
        p = (repo_root / r).resolve()
        if not (p.exists() and p.is_file()):
            continue
        if p.suffix.lower() not in DEFAULT_INCLUDE_EXTS:
            continue
        if not _is_under_any(p, include_roots):
            continue
        files.append(p)
    return files or None

def parse_changed_files_arg(repo_root: Path, include_roots: List[Path], changed_files_csv: str) -> Optional[List[Path]]:
    if not changed_files_csv.strip():
        return None
    rels = [x.strip() for x in changed_files_csv.split(",") if x.strip()]
    files: List[Path] = []
    for r in rels:
        p = (repo_root / r).resolve()
        if not (p.exists() and p.is_file()):
            continue
        if p.suffix.lower() not in DEFAULT_INCLUDE_EXTS:
            continue
        if not _is_under_any(p, include_roots):
            continue
        files.append(p)
    return files or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default=os.getenv("GIT_BRANCH", "main"))
    ap.add_argument("--commit", default=os.getenv("GIT_COMMIT", ""))

    ap.add_argument("--milvus_host", default=os.getenv("MILVUS_HOST", "127.0.0.1"))
    ap.add_argument("--milvus_port", default=os.getenv("MILVUS_PORT", "19530"))
    ap.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "code_chunks"))
    ap.add_argument("--metric", default=os.getenv("MILVUS_METRIC", "IP"))

    ap.add_argument("--embed_model", default=os.getenv("EMBED_MODEL", "krlvi/sentence-t5-base-nlpl-code_search_net"))
    ap.add_argument("--embed_dim", type=int, default=int(os.getenv("EMBED_DIM", "768")))
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--changed_files", default="", help="Comma-separated changed files relative to repo root.")
    ap.add_argument("--full", action="store_true", help="Ingest full repo (ignore changed files).")

    # NEW: restrict ingest roots (default: src)
    ap.add_argument("--include_dirs", default=os.getenv("INGEST_INCLUDE_DIRS", "src"),
                    help="Comma-separated dirs (relative to repo_root) to ingest. Default: src")

    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()

    include_roots = _parse_include_dirs(repo_root, args.include_dirs)
    if not include_roots:
        raise SystemExit(f"No valid include_dirs found under repo_root={repo_root}")

    connections.connect(alias="default", host=args.milvus_host, port=args.milvus_port)

    col = ensure_collection(args.collection, dim=args.embed_dim, metric=args.metric)
    embedder = Embedder(dim=args.embed_dim, model_path=args.embed_model, normalize=True)

    targets = None
    if not args.full:
        targets = parse_changed_files_arg(repo_root, include_roots, args.changed_files) \
                  or parse_changed_files_env(repo_root, include_roots)
    if not targets:
        targets = iter_repo_files(repo_root, include_roots)

    all_chunks: List[Dict[str, Any]] = []
    touched_files: List[str] = []

    for fp in targets:
        rel = str(fp.relative_to(repo_root)).replace("\\", "/")
        chunks = split_code_file(repo_root, fp, repo=args.repo, commit=args.commit)
        if not chunks:
            continue
        for c in chunks:
            c["branch"] = args.branch
        all_chunks.extend(chunks)
        touched_files.append(rel)

    for rel in sorted(set(touched_files)):
        delete_file_chunks(col, args.repo, rel)

    col.flush()
    upsert_chunks(col, all_chunks, embedder=embedder, batch_size=args.batch_size)

    print(f"Done. include_dirs={args.include_dirs} files={len(set(touched_files))} chunks={len(all_chunks)} collection={args.collection}")

if __name__ == "__main__":
    main()
