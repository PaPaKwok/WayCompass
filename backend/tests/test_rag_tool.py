from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents.tools import rag_tool  # noqa: E402


def test_note_keywords_load_destination_rules_from_data_file() -> None:
    """日落规则改为读 JSON，不再写死在代码常量里。"""
    rag_tool._load_retrieval_rules.cache_clear()

    keywords = rag_tool._extract_note_keywords("不想早起，下午看日落拍照", destination="大理")

    assert "日落" in keywords
    assert "洱海" in keywords
    assert "大理180度海景网红打卡地" in keywords
    assert "双廊" not in keywords
    assert "才村" not in keywords


def test_note_keywords_use_current_xiamen_cycling_terms() -> None:
    """骑行扩展词仍指向攻略里存在的环岛路。"""
    rag_tool._load_retrieval_rules.cache_clear()

    keywords = rag_tool._extract_note_keywords("想在海边骑行", destination="厦门")

    assert "骑行" in keywords
    assert "环岛路" in keywords
    assert "洱海生态廊道" not in keywords


def test_retrieval_rule_config_excludes_removed_poi_names() -> None:
    """配置里不许残留攻略中已删掉的旧 POI。"""
    rag_tool._load_retrieval_rules.cache_clear()
    config = rag_tool._load_retrieval_rules()
    serialized = str(config)

    for removed_name in [
        "双廊",
        "才村",
        "龙龛",
        "喜洲古镇",
        "回民街",
        "曾厝垵",
        "洱海生态廊道",
        "蜈支洲岛",
        "第一市场",
        "大熊猫",
        "熊猫",
    ]:
        assert removed_name not in serialized
