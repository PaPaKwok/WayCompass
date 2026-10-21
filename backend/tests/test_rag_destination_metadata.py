from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag import vector_db  # noqa: E402
from scripts import evaluate_rag_retrieval as evaluator  # noqa: E402


def test_loaded_guide_chunks_have_known_destinations() -> None:
    """现有攻略的每个 Chunk 都要带 destination。"""
    chunks = vector_db.load_guide_chunks()

    assert chunks
    assert {chunk["destination"] for chunk in chunks} == {
        "北京", "成都", "大理", "三亚", "厦门", "西安"
    }
    assert all(chunk["destination"] for chunk in chunks)


def test_keyword_fallback_filters_chunks_by_destination(monkeypatch) -> None:
    """Chroma 不可用时，关键词兜底也不能跨城召回。"""
    chunks = [
        {"title": "故宫", "text": "历史建筑", "source": "beijing_guide.md", "destination": "北京"},
        {"title": "大理古城", "text": "历史建筑", "source": "dali_guide.md", "destination": "大理"},
    ]
    monkeypatch.setattr(vector_db, "load_guide_chunks", lambda: chunks)

    results = vector_db._search_guide_chunks_by_keywords(
        query="历史 建筑", top_k=3, destination="北京"
    )

    assert results == [chunks[0]]


def test_chroma_search_filters_by_destination_metadata(monkeypatch) -> None:
    """Chroma 查询要显式按 destination 过滤。"""
    captured: dict[str, object] = {}

    class FakeCollection:
        def count(self) -> int:
            return 1

        def query(self, **kwargs):
            captured.update(kwargs)
            return {
                "documents": [["# 故宫\n明清皇家宫殿建筑群。"]],
                "metadatas": [[
                    {
                        "source": "beijing_guide.md",
                        "title": "故宫",
                        "destination": "北京",
                    }
                ]],
            }

    monkeypatch.setattr(vector_db, "_get_chroma_collection", lambda: FakeCollection())
    monkeypatch.setattr(
        vector_db,
        "_embed_query_with_usage",
        lambda _: ([0.1, 0.2], {"prompt_tokens": 0, "completion_tokens": 0}),
    )

    results, _ = vector_db._search_guide_chunks_by_chroma(
        query="北京历史建筑",
        top_k=3,
        destination="北京",
    )

    assert captured["where"] == {"destination": "北京"}
    assert results[0]["destination"] == "北京"


def test_evaluation_counts_metadata_mismatch_as_cross_destination_pollution(monkeypatch) -> None:
    """污染检测看 metadata，不看文件名里有没有城市。"""
    case = {
        "id": "dali_metadata_check",
        "destination": "大理",
        "preferences": [],
        "top_k": 2,
        "expected_title_keywords": ["大理古城"],
        "required_content_keywords": ["大理古城"],
        "noise_title_keywords": [],
    }
    chunks = [
        {"title": "大理古城", "text": "大理古城历史文化", "source": "guide-a.md", "destination": "大理"},
        {"title": "故宫", "text": "历史建筑", "source": "guide-b.md", "destination": "北京"},
    ]
    monkeypatch.setattr(evaluator, "build_destination_query", lambda **_: ("大理 古城", {}))
    monkeypatch.setattr(
        evaluator,
        "retrieve_travel_guide_chunks",
        lambda **_: (chunks, {"prompt_tokens": 0}, {"prompt_tokens": 0}),
    )

    result = evaluator._evaluate_case(case, {"大理", "北京"})

    assert result["pollution_count"] == 1


def test_evaluation_destinations_are_collected_from_cases() -> None:
    """新加城市由评估 case 自动进入目的地集合。"""
    destinations = evaluator._collect_case_destinations(
        [{"destination": "大理"}, {"destination": "北京"}]
    )

    assert destinations == {"大理", "北京"}
