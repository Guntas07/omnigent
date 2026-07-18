"""Browser coverage for the untitled-to-generated sidebar transition."""

from __future__ import annotations

import uuid

import httpx
from playwright.sync_api import Page, expect


def test_new_chat_placeholder_is_replaced_by_generated_title(
    page: Page,
    seeded_session: tuple[str, str],
) -> None:
    """An untitled row stays usable until its background title arrives."""
    base_url, session_id = seeded_session
    row = page.locator(f'a[href="/c/{session_id}"]')
    generated_title = f"Generated title {uuid.uuid4().hex[:8]}"

    page.goto(f"{base_url}/c/{session_id}")
    expect(row).to_be_visible()
    expect(row).to_contain_text("New Chat")

    response = httpx.patch(
        f"{base_url}/v1/sessions/{session_id}",
        json={"title": generated_title},
        timeout=10.0,
    )
    response.raise_for_status()

    expect(row).to_contain_text(generated_title, timeout=20_000)
    expect(row).not_to_contain_text("New Chat")
