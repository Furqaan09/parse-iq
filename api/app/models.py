from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from sqlmodel import Field, Relationship, SQLModel

# ----------------------
# Enums (SQL friendly)
# ----------------------
class SourceType(str, Enum):
    upload = "upload"
    url = "url"
    email = "email"
    audio = "audio"

class MediaType(str, Enum):
    pdf = "pdf"
    image = "image"
    audio = "audio"
    text = "text"

class Modality(str, Enum):
    text = "text"
    image = "image"

class ValueType(str, Enum):
    date = "date"
    number = "number"
    text = "text"

class Origin(str, Enum):
    docvqa = "docvqa"
    table = "table"
    regex = "regex"
    llm = "llm"

class TaskStatus(str, Enum):
    open = "open"
    done = "done"
    archived = "archived"


# ----------------------------------
# Base model with timestamp fields
# ----------------------------------
class Timestamped(SQLModel):
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

# ----------------
# Document table
# ----------------
class Document(SQLModel, table=True):
    __tablename__ = "documents"
    id: int | None = Field(default=None, primary_key=True)
    user_id: str | None = Field(default=None, index=True)
    title: str = Field(index=True)
    source_type: SourceType = Field(default=SourceType.upload)
    media_type: MediaType = Field(default=MediaType.pdf)
    storage_path: str
    pages: int | None = Field(default=None)
    meta_json: str | None = Field(default=None)

    chunks: list["Chunk"] = Relationship(back_populates="document")
    extractions: list["KVExtraction"] = Relationship(back_populates="document")
    tasks: list["Task"] = Relationship(back_populates="document")

# -------------
# Chunk table
# -------------
class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    modality: Modality = Field(default=Modality.text, index=True) # Is it text or image chunk
    page: int | None = Field(default=None, index=True)  # Page number if applicable
    content_text: str | None = Field(default=None) # Text content if applicable
    bbox: str | None = Field(default=None)  # Bounding box if applicable (for images)
    embedding_key: str | None = Field(default=None, index= True)  # Key to retrieve embedding from vector DB

    document: Document = Relationship(back_populates="chunks")

# --------------------
# KVExtraction table
# --------------------
class KVExtraction(SQLModel, table=True):
    __tablename__ = "kv_extractions"
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    key: str = Field(index=True) # Type of info: "due_date", "total", "merchant"
    value: str = Field(index=True) # Extracted value
    value_type: ValueType = Field(default=ValueType.text, index=True)
    confidence: float | None = Field(default=None, index=True)
    page: int | None = Field(default=None)
    origin: Origin = Field(default=Origin.docvqa)

    document: Document = Relationship(back_populates="extractions")

# ------------
# Task table
# ------------
class Task(SQLModel, table=True):
    __tablename__ = "tasks"
    id: int | None = Field(default=None, primary_key=True)
    document_id: int | None = Field(default=None, foreign_key="documents.id", index=True)
    title: str
    due_at: datetime | None = Field(default=None, index=True)
    location: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    status: TaskStatus = Field(default=TaskStatus.open, index=True)

    document: Optional[Document] = Relationship(back_populates="tasks")
