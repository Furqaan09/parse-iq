from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import DateTime
from sqlmodel import Field, Relationship, SQLModel


# ----------------------
# Enums (SQL friendly)
# ----------------------
class SourceType(str, Enum):
    upload = "upload"
    url = "url"
    email = "email"


class MediaType(str, Enum):
    pdf = "pdf"
    image = "image"
    text = "text"


# ----------------
# Document table
# ----------------
class Document(SQLModel, table=True):
    __tablename__ = "documents"
    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    source_type: SourceType = Field(default=SourceType.upload)
    media_type: MediaType = Field(default=MediaType.pdf)
    storage_path: str
    pages: int | None = Field(default=None)
    meta_json: str | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
        nullable=False,
    )

    chunks: list["Chunk"] = Relationship(back_populates="document")


# -------------
# Chunk table
# -------------
class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    chunk_index: int | None = Field(default=None, index=True)
    page: int | None = Field(default=None, index=True)
    content_text: str | None = Field(default=None)

    document: Document = Relationship(back_populates="chunks")
