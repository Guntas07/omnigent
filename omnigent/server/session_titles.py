"""Optional LLM-backed titles for user-created sessions."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, Protocol

_logger = logging.getLogger(__name__)

TITLE_MAX_LENGTH = 60
TITLE_PROMPT_MAX_LENGTH = 4000

_TITLE_INSTRUCTIONS = """\
Create a short title for the user's task.

The title must be specific, use 3 to 8 words when practical, and be at most
60 characters. Do not answer the request. Do not add quotes, markdown,
punctuation at the end, or generic prefixes such as "Task" or "Help with".
Use only information present in the request.

Return strict JSON matching the supplied schema.
"""

_TITLE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "maxLength": TITLE_MAX_LENGTH,
        }
    },
    "required": ["title"],
    "additionalProperties": False,
}


class SessionTitleGenerator(Protocol):
    """Generate a human-readable title from a user's request."""

    async def generate(
        self,
        prompt: str | None,
        attachments: Sequence[dict[str, Any]] = (),
    ) -> str | None:
        """Return a generated title, or ``None`` to request caller fallback."""


class LLMSessionTitleGenerator:
    """Generate session titles with a server-configured policy LLM client."""

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def generate(
        self,
        prompt: str | None,
        attachments: Sequence[dict[str, Any]] = (),
    ) -> str | None:
        """Generate and validate a title, returning ``None`` on rejection."""
        try:
            content: list[dict[str, Any]] = [
                {
                    "type": "input_text",
                    "text": (
                        prompt[:TITLE_PROMPT_MAX_LENGTH]
                        if prompt
                        else "The user provided an attachment without additional text."
                    ),
                }
            ]
            content.extend(dict(attachment) for attachment in attachments)
            response = await self._llm.create(
                instructions=_TITLE_INSTRUCTIONS,
                input=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "session_title",
                        "strict": True,
                        "schema": _TITLE_SCHEMA,
                    }
                },
            )
            payload = json.loads(_response_text(response))
            title = payload.get("title")
            return _normalize_generated_title(title)
        except Exception:  # noqa: BLE001 -- title generation must never block a turn
            _logger.warning(
                "Session title generation failed; requesting caller fallback",
                exc_info=True,
            )
            return None


def _response_text(response: Any) -> str:
    """Extract the first text block from a Responses-style result."""
    for item in getattr(response, "output", []):
        for block in getattr(item, "content", []):
            text = getattr(block, "text", None)
            if isinstance(text, str):
                return text
    return ""


def _normalize_generated_title(value: Any) -> str | None:
    """Collapse whitespace and enforce the persisted title length limit."""
    if not isinstance(value, str):
        return None
    title = " ".join(value.split()).strip("\"'` ")
    if not title:
        return None
    if len(title) > TITLE_MAX_LENGTH:
        title = title[: TITLE_MAX_LENGTH - 1].rstrip() + "…"
    return title
