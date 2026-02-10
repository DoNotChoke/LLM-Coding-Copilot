from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from typing import List, Dict, Any, Optional


def ensure_collection(name: str, dim: int, metric: str = "IP") -> Collection:
    if utility.has_collection(name):
        return Collection(name)

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),

        FieldSchema(name="repo", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="branch", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="commit", dtype=DataType.VARCHAR, max_length=64),

        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=32),

        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(name="chunk_hash", dtype=DataType.VARCHAR, max_length=64),

        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Code chunks for IDE autocomplete RAG")
    col = Collection(name, schema=schema)

    index_params = {
        "index_type": "HNSW",
        "metric_type": metric,
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index(field_name="embedding", index_params=index_params)

    col.create_index("repo")
    col.create_index("file_path")
    col.create_index("language")

    return col


def delete_file_chunks(col: Collection, repo: str, file_path: str):
    expr = f'repo == "{repo}" && file_path == "{file_path}"'
    col.delete(expr)


def upsert_chunks(col: Collection, chunks: List[Dict[str, Any]], embedder, batch_size: int = 128):
    if not chunks:
        return

    col.load()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [x["text"] for x in batch]
        vecs = embedder.embed_batch(texts)

        data = [
            [x["pk"] for x in batch],
            [x["repo"] for x in batch],
            [x.get("branch", "") for x in batch],
            [x["commit"] for x in batch],
            [x["file_path"] for x in batch],
            [x["language"] for x in batch],
            [int(x["chunk_index"]) for x in batch],
            [x["chunk_hash"] for x in batch],
            [x["text"] for x in batch],
            vecs,
        ]
        col.insert(data)

    col.flush()

def search_similar_chunks(
        col: Collection,
        query_vec: List[float],
        top_k: int = 10,
        threshold: float = 0.5,
        metric: str = "IP",
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        language: Optional[str] = None,
        exclude_file_path: Optional[str] = None,
        include_file_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    col.load()
    filters = []
    if repo:
        filters.append(f'repo == "{repo}"')
    if branch:
        filters.append(f'branch == "{branch}"')
    if language:
        filters.append(f'language == "{language}"')
    if exclude_file_path:
        filters.append(f'file_path != "{exclude_file_path}"')
    if include_file_paths:
        ors = " || ".join([f'file_path == "{p}"' for p in include_file_paths])
        filters.append(f"({ors})")
    expr = " && ".join(filters) if filters else None

    search_params = {"metric_type": metric, "params": {"ef": 128}}

    res = col.search(
        data=[query_vec],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=[
            "repo", "branch", "commit",
            "file_path", "language",
            "chunk_index", "chunk_hash",
            "text",
        ],
    )

    hits = res[0] if res else []
    if not hits:
        return []
    
    out: List[Dict[str, Any]] = []

    for h in hits:
        raw = float(h.distance)

        if metric.upper() == "L2":
            score = -raw
        else:
            score = raw

        if score < float(threshold):
            continue
            
        ent = h.entity

        out.append({
            "pk": h.id,
            "score": score,
            "repo": ent.get("repo"),
            "branch": ent.get("branch"),
            "commit": ent.get("commit"),
            "file_path": ent.get("file_path"),
            "language": ent.get("language"),
            "chunk_index": ent.get("chunk_index"),
            "chunk_hash": ent.get("chunk_hash"),
            "text": ent.get("text"),
        })
    
    if not out:
        return []

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:5]


def embed_and_search(
    col: Collection,
    embedder,
    query_text: str,
    threshold: float = 0.35,
    metric: str = "IP",
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    language: Optional[str] = None,
    exclude_file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    vec = embedder.embed_batch([query_text])[0]
    return search_similar_chunks(
        col=col,
        query_vec=vec,
        threshold=threshold,
        metric=metric,
        repo=repo,
        branch=branch,
        language=language,
        exclude_file_path=exclude_file_path,
    )


milvus_host = "127.0.0.1"
milvus_port = "19530"
milvus_collection = "code_chunks"
milvus_metric = "IP"

col = ensure_collection(milvus_collection, dim=768, metric=milvus_metric)
col.load()
