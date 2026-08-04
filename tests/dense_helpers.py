import json
import shutil
from pathlib import Path

import numpy as np

from medclaim.corpus.scifact_corpus import corpus_content_hash
from medclaim.retrieval.dense import build_dense_index

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bm25_corpus"
MODEL_ID = "fake/semantic-v1"
MODEL_REVISION = "test-rev"


class FakeEmbedder:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    dimension = 4
    device = "cpu"

    def __init__(self):
        self.calls = []

    def encode(
        self,
        texts,
        *,
        batch_size,
        normalize_embeddings,
        show_progress_bar,
    ):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "hydroxychloroquine" in lowered or "viral" in lowered:
                vectors.append([3.0, 0.0, 0.0, 0.0])
            elif any(term in lowered for term in ("vitamin", "immune", "respiratory")):
                vectors.append([0.0, 4.0, 0.0, 0.0])
            elif "aspirin" in lowered or "pain" in lowered:
                vectors.append([0.0, 0.0, 5.0, 0.0])
            elif "shared" in lowered or "scientific" in lowered:
                vectors.append([0.0, 0.0, 0.0, 6.0])
            else:
                vectors.append([1.0, 1.0, 1.0, 1.0])
        return np.asarray(vectors, dtype=np.float64)


def read_passages(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_passages(path, passages):
    path.write_text(
        "".join(json.dumps(passage, separators=(",", ":")) + "\n" for passage in passages),
        encoding="utf-8",
    )


def copy_corpus(destination):
    destination.mkdir(parents=True)
    shutil.copy2(FIXTURES / "passages.jsonl", destination / "passages.jsonl")
    shutil.copy2(FIXTURES / "manifest.json", destination / "manifest.json")


def update_corpus_manifest(corpus_dir, passages):
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["passage_count"] = len(passages)
    manifest["content_hash"] = corpus_content_hash(passages)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def build_fake_dense_index(tmp_path, version="dense-v1", embedder=None):
    corpus_dir = tmp_path / "corpus"
    copy_corpus(corpus_dir)
    selected_embedder = embedder or FakeEmbedder()
    index_dir = build_dense_index(
        corpus_dir,
        tmp_path / "indexes",
        version,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        batch_size=2,
        device="cpu",
        embedder=selected_embedder,
        show_progress_bar=False,
    )
    return corpus_dir, index_dir, selected_embedder
