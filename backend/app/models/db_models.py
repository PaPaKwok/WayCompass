from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.config import Base


class TripRecord(Base):
    """本版本只用到这一张行程表。"""

    __tablename__ = "trip_records"

    # 自增主键，仅库内使用
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # 对外暴露的行程编号
    trip_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    destination: Mapped[str] = mapped_column(String(100))
    summary: Mapped[str] = mapped_column(Text)
    # 整份行程以 JSON 文本落库
    itinerary_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
