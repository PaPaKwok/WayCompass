from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import (
    AMAP_API_KEY,
    AMAP_BASE_URL,
    AMAP_TIMEOUT_SECONDS,
    REDIS_WEATHER_TTL_SECONDS,
)
from app.services.cache_service import get_cached_json, set_cached_json
from app.services.map_service import geocode_address


logger = logging.getLogger(__name__)


def _ensure_amap_api_key() -> None:
    """校验天气接口所需的高德 Key。"""
    if not AMAP_API_KEY:
        raise RuntimeError("当前环境未配置 AMAP_API_KEY，无法调用天气服务。")


def _build_client() -> httpx.Client:
    """构造高德天气的 HTTP 客户端。"""
    return httpx.Client(timeout=AMAP_TIMEOUT_SECONDS)


def _request_amap_weather(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """请求天气接口并解析响应。"""
    _ensure_amap_api_key()

    request_params = {
        "key": AMAP_API_KEY,
        **params,
    }

    with _build_client() as client:
        response = client.get(f"{AMAP_BASE_URL}{path}", params=request_params)
        response.raise_for_status()
        payload = response.json()

    if payload.get("status") != "1":
        info = payload.get("info", "未知错误")
        raise RuntimeError(f"高德天气接口调用失败：{info}")

    return payload


def _normalize_cache_text(value: str | None) -> str:
    """标准化文本，保证缓存键稳定。"""
    if value is None:
        return ""
    return value.strip().lower()


def get_weather_forecast(city: str) -> dict[str, Any]:
    """取指定城市的未来天气。"""
    cache_key = f"weather:forecast:{_normalize_cache_text(city)}"
    cached_value = get_cached_json(cache_key)
    if cached_value is not None:
        logger.info("weather cache hit: city=%s", city)
        return cached_value
    logger.info("weather cache miss: city=%s", city)

    geocode = geocode_address(city, city=city)
    city_code = geocode.get("adcode") if geocode is not None else city

    payload = _request_amap_weather(
        "/weather/weatherInfo",
        {
            "city": city_code or city,
            "extensions": "all",
        },
    )

    forecasts = payload.get("forecasts", [])
    if not forecasts:
        raise RuntimeError("未获取到天气预报结果。")

    first = forecasts[0]
    casts = first.get("casts", [])

    days = [
        {
            "date": cast.get("date"),
            "week": cast.get("week"),
            "day_weather": cast.get("dayweather"),
            "night_weather": cast.get("nightweather"),
            "day_temp": cast.get("daytemp"),
            "night_temp": cast.get("nighttemp"),
            "day_wind": cast.get("daywind"),
            "night_wind": cast.get("nightwind"),
        }
        for cast in casts
    ]

    result = {
        "city": first.get("city") or city,
        "province": first.get("province"),
        "adcode": first.get("adcode"),
        "report_time": first.get("reporttime"),
        "days": days,
    }
    set_cached_json(cache_key, result, expire_seconds=REDIS_WEATHER_TTL_SECONDS)
    return result
