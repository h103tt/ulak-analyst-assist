"""Gemini API key discovery.

``.env`` may hold a single ``GEMINI_API_KEY``, or a numbered pool
(``GEMINI_API_KEY1`` .. ``GEMINI_API_KEYN``). The pool exists because the
free tier caps embedding calls per *project* per day, so spreading a large
knowledge-base ingest across keys from several projects is the only way to
get through it in one sitting (see kb_ingest.py).
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv()

_NUMBERED_KEY = re.compile(r"^GEMINI_API_KEY(\d+)$")


def all_keys() -> list[str]:
    """Every configured key, in use order: the bare ``GEMINI_API_KEY`` first
    when set, then ``GEMINI_API_KEY<N>`` sorted numerically.

    The numeric sort matters -- sorting these names as strings would put
    ``GEMINI_API_KEY10`` ahead of ``GEMINI_API_KEY2``.
    """
    keys: list[str] = []

    bare = os.getenv("GEMINI_API_KEY")
    if bare:
        keys.append(bare)

    numbered: list[tuple[int, str]] = []
    for name, value in os.environ.items():
        match = _NUMBERED_KEY.match(name)
        if match and value:
            numbered.append((int(match.group(1)), value))
    keys.extend(value for _, value in sorted(numbered))

    seen: set[str] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def primary_key() -> str:
    """The first configured key, unvalidated. Raises if none is set.

    Prefer ``working_key()`` for anything that actually calls the API --
    the first key in ``.env`` isn't guaranteed to be live (keys get
    exhausted or swapped out), and this doesn't check.
    """
    keys = all_keys()
    if not keys:
        raise EnvironmentError(
            "No Gemini API key found. Set GEMINI_API_KEY, or a numbered pool "
            "GEMINI_API_KEY1..N, in your .env file."
        )
    return keys[0]


_working_key_cache: dict[str, str] = {}


def working_key(probe, purpose: str = "default") -> str:
    """The first configured key that actually works, found by calling
    ``probe(key)`` (expected to raise on failure) down the list, and cached
    per ``purpose`` for the rest of the process so this only costs one real
    API call per key tried, once.

    ``purpose`` matters: a key exhausted for chat generation may still have
    embedding quota left, and vice versa (they're billed separately), so
    chat and embeddings should probe -- and cache -- independently. Pass a
    distinct ``purpose`` string per use case (e.g. "chat", "embed").
    """
    if purpose in _working_key_cache:
        return _working_key_cache[purpose]

    keys = all_keys()
    if not keys:
        raise EnvironmentError(
            "No Gemini API key found. Set GEMINI_API_KEY, or a numbered pool "
            "GEMINI_API_KEY1..N, in your .env file."
        )

    last_error: Exception | None = None
    for key in keys:
        try:
            probe(key)
        except Exception as e:  # noqa: BLE001 - any failure disqualifies the key
            last_error = e
            continue
        _working_key_cache[purpose] = key
        return key

    raise EnvironmentError(
        f"None of the {len(keys)} configured Gemini API key(s) work for "
        f"'{purpose}'. Last error: {last_error}"
    ) from last_error
