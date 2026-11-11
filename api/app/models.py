from datetime import UTC, datetime
from enum import Enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Computed, DateTime, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlmodel import Field, Relationship, SQLModel

EMBEDDING_DIMS = 1536  # text-embedding-3-small


# ----------------------
# Enums (SQL friendly)
# ----------------------
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
    media_type: MediaType = Field(default=MediaType.pdf)
    storage_path: str
    pages: int | None = Field(default=None)
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
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    chunk_index: int | None = Field(default=None, index=True)
    page: int | None = Field(default=None, index=True)
    content_text: str | None = Field(default=None)
    # Semantic search
    embedding: list[float] | None = Field(default=None, sa_type=Vector(EMBEDDING_DIMS))
    # Full-text search: automatically updated by Postgres whenever content_text changes
    search_vector: str | None = Field(
        default=None,
        sa_column=Column(
            TSVECTOR,
            Computed("to_tsvector('english', coalesce(content_text, ''))", persisted=True),
            nullable=True,
        ),
    )

    document: Document = Relationship(back_populates="chunks")
