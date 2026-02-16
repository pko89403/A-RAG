"""Pydantic models for chat history."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class Reference(BaseModel):
    """Reference document returned from RAG search."""

    id: str
    title: str
    type: Literal["document", "image"] = "document"
    content: str | None = None
    source: str | None = None
    score: float | None = None
    used_chunks: list[str] | None = Field(default=None, alias="usedChunks")
    used_in_answer: bool | None = Field(default=None, alias="usedInAnswer")

    class Config:
        populate_by_name = True


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    references: list[Reference] | None = None

    class Config:
        populate_by_name = True


class Conversation(BaseModel):
    """A conversation containing multiple messages."""

    id: str
    user_id: str = Field(alias="userId")
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=datetime.utcnow, alias="updatedAt")
    turn_count: int = Field(default=0, alias="turnCount")
    messages: list[ChatMessage] = Field(default_factory=list)

    class Config:
        populate_by_name = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "userId": self.user_id,
            "title": self.title,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "turnCount": self.turn_count,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "references": [
                        ref.model_dump(by_alias=True) for ref in msg.references
                    ]
                    if msg.references
                    else None,
                }
                for msg in self.messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create from a dictionary (e.g. loaded from JSON)."""
        messages = []
        for msg_data in data.get("messages", []):
            refs = None
            if msg_data.get("references"):
                refs = [Reference(**ref) for ref in msg_data["references"]]
            messages.append(
                ChatMessage(
                    id=msg_data["id"],
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"])
                    if msg_data.get("timestamp")
                    else datetime.utcnow(),
                    references=refs,
                )
            )

        return cls(
            id=data["id"],
            userId=data["userId"],
            title=data["title"],
            createdAt=datetime.fromisoformat(data["createdAt"])
            if data.get("createdAt")
            else datetime.utcnow(),
            updatedAt=datetime.fromisoformat(data["updatedAt"])
            if data.get("updatedAt")
            else datetime.utcnow(),
            turnCount=data.get("turnCount", 0),
            messages=messages,
        )


class ConversationSummary(BaseModel):
    """Summary of a conversation for list view."""

    id: str
    title: str
    timestamp: str  # Relative time string for display
    created_at: datetime = Field(alias="createdAt")
    turn_count: int = Field(alias="turnCount")

    class Config:
        populate_by_name = True
