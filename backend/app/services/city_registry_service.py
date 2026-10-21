"""
目的地名称规范化与城市覆盖等级登记。

注册表目前只收录已有本地攻略的城市；接入地图解析后，
未登记的城市会在这里被判成 dynamic 或 insufficient_data。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.rag.guide_catalog import GUIDE_DESTINATIONS


class CityCoverageTier(StrEnum):
    """城市当前所处的覆盖层级枚举。"""

    CURATED = "curated"
    DYNAMIC = "dynamic"
    INSUFFICIENT_DATA = "insufficient_data"
    UNREGISTERED = "unregistered"


class CityKnowledgeStatus(StrEnum):
    """该城市知识库的可用状态。"""

    READY = "ready"
    UNREGISTERED = "unregistered"


@dataclass(frozen=True)
class CityRegistryEntry:
    """已收录城市在注册表中的静态描述。"""

    city: str
    aliases: tuple[str, ...]
    guide_sources: tuple[str, ...]
    tier: CityCoverageTier = CityCoverageTier.CURATED
    knowledge_status: CityKnowledgeStatus = CityKnowledgeStatus.READY


@dataclass(frozen=True)
class CityLookupResult:
    """目的地查询返回的注册信息快照。"""

    city: str
    tier: CityCoverageTier
    knowledge_status: CityKnowledgeStatus
    entry: CityRegistryEntry | None = None


def normalize_city_name(destination: str) -> str:
    """统一空白与常见“市”后缀，得到标准城市名。"""
    normalized = "".join(destination.split())
    if not normalized:
        raise ValueError("目的地不能为空")

    if (
        normalized.endswith("市")
        and not normalized.endswith("城市")
        and len(normalized) > 1
    ):
        normalized = normalized[:-1]

    return normalized


def _build_curated_city_registry() -> dict[str, CityRegistryEntry]:
    """由现有攻略目录推导出过渡期的注册表。"""
    sources_by_city: dict[str, list[str]] = {}
    for source, city in GUIDE_DESTINATIONS.items():
        sources_by_city.setdefault(city, []).append(source)

    registry: dict[str, CityRegistryEntry] = {}
    for city, sources in sources_by_city.items():
        registry[city] = CityRegistryEntry(
            city=city,
            aliases=(city, f"{city}市"),
            guide_sources=tuple(sorted(sources)),
        )
    return registry


CURATED_CITY_REGISTRY = _build_curated_city_registry()
_CITY_ALIASES = {
    alias: entry.city
    for entry in CURATED_CITY_REGISTRY.values()
    for alias in entry.aliases
}


def lookup_city(destination: str) -> CityLookupResult:
    """判断某个目的地是否已有可检索的本地攻略。"""
    normalized = normalize_city_name(destination)
    canonical_city = _CITY_ALIASES.get(normalized, normalized)
    entry = CURATED_CITY_REGISTRY.get(canonical_city)

    if entry is not None:
        return CityLookupResult(
            city=entry.city,
            tier=entry.tier,
            knowledge_status=entry.knowledge_status,
            entry=entry,
        )

    return CityLookupResult(
        city=canonical_city,
        tier=CityCoverageTier.UNREGISTERED,
        knowledge_status=CityKnowledgeStatus.UNREGISTERED,
    )
