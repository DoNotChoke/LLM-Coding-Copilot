from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from typing import List, Dict, Any


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