"""Tests for optional LLM-generated session titles."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest

from omnigent.entities import Conversation, MessageData
from omnigent.server.routes import sessions as sessions_module
from omnigent.server.session_titles import LLMSessionTitleGenerator
from omnigent.stores.conversation_store import ConversationStore


class _FakeLLM:
    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text=self.result)])]
        )


class _FakeTitleGenerator:
    def __init__(self, title: str | None) -> None:
        self.title = title
        self.prompts: list[str] = []
        self.attachments: list[tuple[dict[str, object], ...]] = []

    async def generate(
        self,
        prompt: str | None,
        attachments: Sequence[dict[str, object]] = (),
    ) -> str | None:
        if prompt is not None:
            self.prompts.append(prompt)
        self.attachments.append(tuple(attachments))
        return self.title


class _FakeConversationStore:
    def __init__(self, items: list[tuple[str, str]] | None = None) -> None:
        self.saved_title: str | None = None
        self.items = [
            SimpleNamespace(
                id=item_id,
                type="message",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": prompt}],
                ),
            )
            for item_id, prompt in (items or [("item_1", "Initial prompt")])
        ]

    def list_items(
        self,
        conversation_id: str,
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
        type: str | None = None,
    ) -> object:
        assert conversation_id == "session_1"
        del before, order, type
        start = 0
        if after is not None:
            start = next(i + 1 for i, item in enumerate(self.items) if item.id == after)
        data = self.items[start : start + limit]
        return SimpleNamespace(
            data=data,
            last_id=data[-1].id if data else None,
            has_more=start + limit < len(self.items),
        )

    def set_title_if_missing(
        self,
        conversation_id: str,
        title: str,
    ) -> bool:
        assert conversation_id == "session_1"
        if self.saved_title is not None:
            return False
        self.saved_title = title
        return True

    def get_conversation(self, conversation_id: str) -> object:
        assert conversation_id == "session_1"
        return SimpleNamespace(title=self.saved_title)


class _BlockingTitleGenerator(_FakeTitleGenerator):
    def __init__(self, title: str | None) -> None:
        super().__init__(title)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(
        self,
        prompt: str | None,
        attachments: Sequence[dict[str, object]] = (),
    ) -> str | None:
        if prompt is not None:
            self.prompts.append(prompt)
        self.attachments.append(tuple(attachments))
        self.started.set()
        await self.release.wait()
        return self.title


class _SequencedTitleGenerator(_FakeTitleGenerator):
    def __init__(self, results: list[str | None | Exception]) -> None:
        super().__init__(None)
        self.results = results
        self.attempt_finished = asyncio.Event()

    async def generate(
        self,
        prompt: str | None,
        attachments: Sequence[dict[str, object]] = (),
    ) -> str | None:
        if prompt is not None:
            self.prompts.append(prompt)
        self.attachments.append(tuple(attachments))
        result = self.results[len(self.attachments) - 1]
        self.attempt_finished.set()
        if isinstance(result, Exception):
            raise result
        return result


async def _finish_title_tasks() -> None:
    tasks = list(sessions_module._session_title_tasks)
    if tasks:
        await asyncio.gather(*tasks)


@pytest.fixture(autouse=True)
def _reset_title_inflight() -> None:
    sessions_module._session_title_inflight_ids.clear()


@pytest.mark.asyncio
async def test_llm_title_generator_requests_structured_concise_title() -> None:
    llm = _FakeLLM('{"title":"Diagnose OAuth Callback Failure"}')
    generator = LLMSessionTitleGenerator(llm)

    title = await generator.generate(
        "Our OAuth callback fails after login. Please inspect the redirect handling."
    )

    assert title == "Diagnose OAuth Callback Failure"
    call = llm.calls[0]
    assert call["input"][0]["content"][0]["text"].startswith("Our OAuth")  # type: ignore[index]
    assert call["text"]["format"]["schema"]["properties"]["title"]["maxLength"] == 60  # type: ignore[index]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        RuntimeError("provider unavailable"),
        "not json",
        '{"title":"   "}',
        '{"unexpected":"value"}',
    ],
)
async def test_llm_title_generator_fails_open(result: str | Exception) -> None:
    generator = LLMSessionTitleGenerator(_FakeLLM(result))

    assert await generator.generate("A user request") is None


@pytest.mark.asyncio
async def test_llm_title_generator_normalizes_and_limits_output() -> None:
    generator = LLMSessionTitleGenerator(_FakeLLM('{"title":"  `A   better   title`  "}'))
    assert await generator.generate("request") == "A better title"

    long_title = "x" * 80
    generator = LLMSessionTitleGenerator(_FakeLLM(f'{{"title":"{long_title}"}}'))
    title = await generator.generate("request")
    assert title == "x" * 59 + "…"


@pytest.mark.asyncio
async def test_llm_title_generator_sends_images_to_the_model() -> None:
    llm = _FakeLLM('{"title":"Golden Retriever at Beach"}')
    generator = LLMSessionTitleGenerator(llm)

    title = await generator.generate(
        None,
        [{"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="}],
    )

    assert title == "Golden Retriever at Beach"
    content = llm.calls[0]["input"][0]["content"]  # type: ignore[index]
    assert content[1] == {  # type: ignore[index]
        "type": "input_image",
        "image_url": "data:image/png;base64,aW1hZ2U=",
    }


@pytest.mark.asyncio
async def test_title_seed_stays_untitled_until_generator_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _BlockingTitleGenerator("Repair OAuth Redirect Handling")
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    prompt = "Investigate the OAuth redirect failure " + "with detailed context " * 10
    raw_store = _FakeConversationStore([("item_1", prompt)])
    store = cast(ConversationStore, raw_store)

    await sessions_module._seed_missing_title(
        conv,
        store,
    )

    await generated.started.wait()
    assert len(generated.prompts[0]) > 60
    assert raw_store.saved_title is None
    assert conv.title is None

    generated.release.set()
    await _finish_title_tasks()
    assert raw_store.saved_title == "Repair OAuth Redirect Handling"


@pytest.mark.asyncio
async def test_generated_title_does_not_overwrite_manual_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _BlockingTitleGenerator("Generated title")
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore()

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await generated.started.wait()

    raw_store.saved_title = "My manual title"
    generated.release.set()
    await _finish_title_tasks()

    assert raw_store.saved_title == "My manual title"


@pytest.mark.asyncio
async def test_rejected_generation_retries_then_uses_original_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _FakeTitleGenerator(None)
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    monkeypatch.setattr(sessions_module, "_SESSION_TITLE_RETRY_DELAY_SECONDS", 0)
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore(
        [("item_1", "Keep this deterministic"), ("item_2", "A later prompt")]
    )
    store = cast(ConversationStore, raw_store)

    await sessions_module._seed_missing_title(
        conv,
        store,
    )

    await _finish_title_tasks()
    assert raw_store.saved_title == "Keep this deterministic"
    assert generated.prompts == ["Keep this deterministic"] * 3


@pytest.mark.asyncio
async def test_title_generation_retries_exception_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _SequencedTitleGenerator(
        [RuntimeError("temporary provider failure"), "Recovered generated title"]
    )
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    monkeypatch.setattr(sessions_module, "_SESSION_TITLE_RETRY_DELAY_SECONDS", 0)
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore([("item_1", "Investigate retry behavior")])

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await _finish_title_tasks()

    assert generated.prompts == ["Investigate retry behavior"] * 2
    assert raw_store.saved_title == "Recovered generated title"


@pytest.mark.asyncio
async def test_manual_rename_during_retry_prevents_generation_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _SequencedTitleGenerator([None, "Must not be used"])
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    monkeypatch.setattr(sessions_module, "_SESSION_TITLE_RETRY_DELAY_SECONDS", 0.05)
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore([("item_1", "Investigate manual renaming")])

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await generated.attempt_finished.wait()
    raw_store.saved_title = "My manual title"
    await _finish_title_tasks()

    assert generated.prompts == ["Investigate manual renaming"]
    assert raw_store.saved_title == "My manual title"


@pytest.mark.asyncio
async def test_concurrent_seed_requests_use_oldest_persisted_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _FakeTitleGenerator("Oldest request title")
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore(
        [("item_1", "Oldest persisted request"), ("item_2", "Later request")]
    )
    store = cast(ConversationStore, raw_store)

    await asyncio.gather(
        sessions_module._seed_missing_title(conv, store),
        sessions_module._seed_missing_title(conv, store),
    )
    await _finish_title_tasks()

    assert generated.prompts == ["Oldest persisted request"]
    assert raw_store.saved_title == "Oldest request title"


@pytest.mark.asyncio
async def test_image_only_request_is_titled_from_resolved_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _FakeTitleGenerator("Golden Retriever at Beach")
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    monkeypatch.setattr(
        sessions_module,
        "get_file_store",
        lambda: SimpleNamespace(
            get=lambda file_id: SimpleNamespace(
                id=file_id,
                session_id="session_1",
                content_type="image/png",
                filename="dog.png",
            )
        ),
    )
    monkeypatch.setattr(
        sessions_module,
        "get_artifact_store",
        lambda: SimpleNamespace(get=lambda file_id: b"image"),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore()
    raw_store.items = [
        SimpleNamespace(
            id="item_1",
            type="message",
            data=MessageData(
                role="user",
                content=[{"type": "input_image", "file_id": "file_1", "filename": "dog.png"}],
            ),
        )
    ]

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await _finish_title_tasks()

    assert raw_store.saved_title == "Golden Retriever at Beach"
    assert generated.prompts == []
    assert generated.attachments == [
        (
            {
                "type": "input_image",
                "filename": "dog.png",
                "image_url": "data:image/png;base64,aW1hZ2U=",
            },
        )
    ]


@pytest.mark.asyncio
async def test_text_and_pdf_are_both_sent_to_title_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _FakeTitleGenerator("Hutchinson Farms Case Study")
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    monkeypatch.setattr(
        sessions_module,
        "get_file_store",
        lambda: SimpleNamespace(
            get=lambda file_id: SimpleNamespace(
                id=file_id,
                session_id="session_1",
                content_type="application/pdf",
                filename="case-study.pdf",
            )
        ),
    )
    monkeypatch.setattr(
        sessions_module,
        "get_artifact_store",
        lambda: SimpleNamespace(get=lambda file_id: b"%PDF"),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore()
    raw_store.items = [
        SimpleNamespace(
            id="item_1",
            type="message",
            data=MessageData(
                role="user",
                content=[
                    {"type": "input_text", "text": "a"},
                    {
                        "type": "input_file",
                        "file_id": "file_1",
                        "filename": "case-study.pdf",
                    },
                ],
            ),
        )
    ]

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await _finish_title_tasks()

    assert raw_store.saved_title == "Hutchinson Farms Case Study"
    assert generated.prompts == ["a"]
    assert generated.attachments == [
        (
            {
                "type": "input_file",
                "filename": "case-study.pdf",
                "file_data": "data:application/pdf;base64,JVBERg==",
            },
        )
    ]


@pytest.mark.asyncio
async def test_without_generator_image_only_request_does_not_block_later_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=None),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore()
    raw_store.items = [
        SimpleNamespace(
            id="item_1",
            type="message",
            data=MessageData(
                role="user",
                content=[{"type": "input_image", "file_id": "file_1"}],
            ),
        ),
        SimpleNamespace(
            id="item_2",
            type="message",
            data=MessageData(
                role="user",
                content=[{"type": "input_text", "text": "Explain this architecture"}],
            ),
        ),
    ]

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )

    assert raw_store.saved_title == "Explain this architecture"


@pytest.mark.asyncio
async def test_title_seed_uses_original_title_without_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=None),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    prompt = "Explain how to migrate this application without downtime " * 3
    raw_store = _FakeConversationStore([("item_1", prompt)])

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await _finish_title_tasks()

    assert raw_store.saved_title == prompt[:59].rstrip() + "…"
    assert conv.title == raw_store.saved_title


@pytest.mark.asyncio
async def test_stale_untitled_entity_does_not_trigger_duplicate_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _FakeTitleGenerator("Duplicate title")
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore()
    raw_store.saved_title = "Already generated"

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await _finish_title_tasks()

    assert generated.prompts == []
    assert raw_store.saved_title == "Already generated"


@pytest.mark.asyncio
async def test_shutdown_allows_generated_title_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _BlockingTitleGenerator("Finished during shutdown")
    monkeypatch.setattr(
        sessions_module,
        "get_caps",
        lambda: SimpleNamespace(session_title_generator=generated),
    )
    monkeypatch.setattr(sessions_module, "_SESSION_TITLE_SHUTDOWN_TIMEOUT_SECONDS", 0.1)
    conv = cast(Conversation, SimpleNamespace(id="session_1", title=None))
    raw_store = _FakeConversationStore([("item_1", "Keep deterministic fallback")])

    await sessions_module._seed_missing_title(
        conv,
        cast(ConversationStore, raw_store),
    )
    await generated.started.wait()

    async def _release_generator() -> None:
        await asyncio.sleep(0.01)
        generated.release.set()

    release_task = asyncio.create_task(_release_generator())
    await sessions_module.cancel_session_title_tasks()
    await release_task

    assert raw_store.saved_title == "Finished during shutdown"
    assert not sessions_module._session_title_tasks


def test_hidden_meta_message_is_not_a_title_candidate() -> None:
    item = SimpleNamespace(
        type="message",
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": "Internal skill instructions"}],
            is_meta=True,
        ),
    )

    assert sessions_module._title_content_from_item(item) == []
