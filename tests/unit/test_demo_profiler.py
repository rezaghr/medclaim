import importlib.util
from pathlib import Path


def test_demo_profile_is_honest_and_complete():
    path = Path(__file__).resolve().parents[2] / "scripts" / "profile_demo_pipeline.py"
    spec = importlib.util.spec_from_file_location("profile_demo_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.profile(3, 1)
    assert report["measurement_mode"] == "synthetic_fake_components"
    assert "not external models" in report["measurement_warning"]
    assert all(not stage["target_demonstrated"] for stage in report["stages"].values())
    assert set(report["stages"]) == {
        "validation",
        "bm25",
        "dense_retrieval",
        "fusion_metadata",
        "reranking",
        "llm_verification",
        "total",
    }
