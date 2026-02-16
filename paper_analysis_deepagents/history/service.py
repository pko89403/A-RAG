"""Local JSON file-based chat history service."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from paper_analysis_deepagents.history.models import (
    ChatMessage,
    Conversation,
    ConversationSummary,
)

logger = logging.getLogger(__name__)


class HistoryServiceError(Exception):
    """Base exception for history service errors."""

    pass


class ConversationNotFoundError(HistoryServiceError):
    """Raised when a conversation is not found."""

    pass


class MaxTurnsExceededError(HistoryServiceError):
    """Raised when max turns limit is exceeded."""

    pass


class HistoryService:
    """Service for managing chat history with local JSON files.

    Storage Structure:
    - Root: data/history/ (configurable)
    - Per-user directory: {root}/{user_id}/
    - Per-conversation file: {root}/{user_id}/{conversation_id}.json

    JSON Schema (same as previous CosmosDB document):
    {
        "id": "conv_123456",
        "userId": "user-abc-def",
        "title": "투자 심의서 검토...",
        "createdAt": "2026-02-06T10:00:00",
        "updatedAt": "2026-02-06T10:30:00",
        "turnCount": 3,
        "messages": [...]
    }
    """

    def __init__(
        self,
        root_dir: str | Path = "data/history",
        max_turns: int = 5,
    ):
        """Initialize local JSON history service.

        Args:
            root_dir: Root directory for storing history JSON files.
            max_turns: Maximum conversation turns allowed.
        """
        self.root_dir = Path(root_dir)
        self.max_turns = max_turns

        # Ensure root directory exists
        self._ensure_dir(self.root_dir)
        logger.info("Local history service initialized: root_dir=%s", self.root_dir)

    def _ensure_dir(self, path: Path) -> None:
        """Create directory if it does not exist."""
        path.mkdir(parents=True, exist_ok=True)

    def _user_dir(self, user_id: str) -> Path:
        """Return path to a user's history directory."""
        return self.root_dir / user_id

    def _conversation_path(self, user_id: str, conversation_id: str) -> Path:
        """Return path to a conversation JSON file."""
        return self._user_dir(user_id) / f"{conversation_id}.json"

    def _read_conversation(self, user_id: str, conversation_id: str) -> Conversation:
        """Read a conversation from disk."""
        path = self._conversation_path(user_id, conversation_id)
        if not path.exists():
            raise ConversationNotFoundError(
                f"Conversation {conversation_id} not found for user {user_id}"
            )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Conversation.from_dict(data)

    def _write_conversation(self, conversation: Conversation) -> None:
        """Write a conversation to disk."""
        user_dir = self._user_dir(conversation.user_id)
        self._ensure_dir(user_dir)
        path = user_dir / f"{conversation.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)

    def list_conversations(self, user_id: str) -> list[ConversationSummary]:
        """List all conversations for a user.

        Args:
            user_id: User ID

        Returns:
            List of conversation summaries, sorted by most recent first
        """
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return []

        summaries: list[ConversationSummary] = []
        for json_file in sorted(
            user_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                created_at = datetime.fromisoformat(data["createdAt"])
                summaries.append(
                    ConversationSummary(
                        id=data["id"],
                        title=data["title"],
                        timestamp=self._get_relative_time(created_at),
                        createdAt=created_at,
                        turnCount=data.get("turnCount", 0),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping malformed history file %s: %s", json_file, e)
                continue

        return summaries

    def get_conversation(self, user_id: str, conversation_id: str) -> Conversation:
        """Get a specific conversation with all messages.

        Args:
            user_id: User ID
            conversation_id: Conversation ID

        Returns:
            Full conversation with messages

        Raises:
            ConversationNotFoundError: If conversation not found
        """
        return self._read_conversation(user_id, conversation_id)

    def create_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str,
        initial_message: ChatMessage | None = None,
    ) -> Conversation:
        """Create a new conversation.

        Args:
            user_id: User ID
            conversation_id: Conversation ID
            title: Conversation title
            initial_message: Optional first message

        Returns:
            Created conversation
        """
        now = datetime.utcnow()
        messages = [initial_message] if initial_message else []

        conversation = Conversation(
            id=conversation_id,
            userId=user_id,
            title=title,
            createdAt=now,
            updatedAt=now,
            turnCount=1 if initial_message and initial_message.role == "user" else 0,
            messages=messages,
        )

        self._write_conversation(conversation)
        logger.info("Created conversation %s for user %s", conversation_id, user_id)

        return conversation

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        message: ChatMessage,
    ) -> Conversation:
        """Add a message to a conversation.

        Args:
            user_id: User ID
            conversation_id: Conversation ID
            message: Message to add

        Returns:
            Updated conversation

        Raises:
            ConversationNotFoundError: If conversation not found
            MaxTurnsExceededError: If max turns exceeded
        """
        conversation = self._read_conversation(user_id, conversation_id)

        # Check turn limit (only count user messages as turns)
        if message.role == "user":
            if conversation.turn_count < self.max_turns:
                conversation.turn_count += 1

        # Add message and update timestamp
        conversation.messages.append(message)
        conversation.updated_at = datetime.utcnow()

        # Update title if first user message
        if message.role == "user" and conversation.turn_count == 1:
            conversation.title = message.content[:30] + (
                "..." if len(message.content) > 30 else ""
            )

        # Write back to disk
        self._write_conversation(conversation)

        logger.info(
            "Added message to conversation %s (turn %d/%d)",
            conversation_id,
            conversation.turn_count,
            self.max_turns,
        )

        return conversation

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        """Delete a conversation.

        Args:
            user_id: User ID
            conversation_id: Conversation ID

        Raises:
            ConversationNotFoundError: If conversation not found
        """
        path = self._conversation_path(user_id, conversation_id)
        if not path.exists():
            raise ConversationNotFoundError(
                f"Conversation {conversation_id} not found for user {user_id}"
            )
        path.unlink()
        logger.info("Deleted conversation %s for user %s", conversation_id, user_id)

    def _get_relative_time(self, dt: datetime) -> str:
        """Convert datetime to relative time string."""
        now = datetime.utcnow()
        diff = now - dt

        minutes = int(diff.total_seconds() / 60)
        hours = int(diff.total_seconds() / 3600)
        days = diff.days

        if minutes < 1:
            return "방금 전"
        if minutes < 60:
            return f"{minutes}분 전"
        if hours < 24:
            return f"{hours}시간 전"
        if days < 7:
            return f"{days}일 전"
        return dt.strftime("%Y-%m-%d")
