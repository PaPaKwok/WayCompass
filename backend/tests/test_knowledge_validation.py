from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag import knowledge_validation  # noqa: E402


def test_current_knowledge_base_is_consistent() -> None:
    """攻略、规则、兜底与评估断言需彼此对得上。"""
    assert knowledge_validation.validate_knowledge_base() == []


def test_validator_rejects_rule_keyword_missing_from_destination_source() -> None:
    """塞进失效扩展词时校验必须报错。"""
    errors: list[str] = []
    knowledge_validation._validate_retrieval_rules(
        {
            "global_rules": [
                {"triggers": ["历史"], "keywords": ["故宫"]},
            ],
            "destinations": {
                "北京": {
                    "rules": [
                        {"triggers": ["不存在"], "keywords": ["不存在的景点"]},
                    ]
                }
            },
        },
        {"北京": "故宫博物院"},
        errors,
    )

    assert errors == ["北京 rules 第 1 条扩展词未命中攻略：不存在的景点"]
