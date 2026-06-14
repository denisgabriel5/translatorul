import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import translate


def _patch_load(monkeypatch):
    # Avoid requiring the real GGUF model file / llama-cpp runtime.
    monkeypatch.setattr(translate, "_load", lambda: None)
    monkeypatch.setattr(translate, "_llm", object())


def test_batch_translation_preserves_count(monkeypatch):
    _patch_load(monkeypatch)
    monkeypatch.setattr(translate, "BATCH_SIZE", 3)
    monkeypatch.setattr(translate, "CONTEXT_CUES", 1)

    def fake_chat(system_prompt, user_prompt, max_tokens):
        return "1. unu\n2. doi\n3. trei"

    monkeypatch.setattr(translate, "_chat", fake_chat)

    texts = ["one", "two", "three"]
    result = translate.translate_text(texts, "ro")

    assert result == ["unu", "doi", "trei"]


def test_mismatched_batch_falls_back_to_per_cue(monkeypatch):
    _patch_load(monkeypatch)
    monkeypatch.setattr(translate, "BATCH_SIZE", 3)
    monkeypatch.setattr(translate, "CONTEXT_CUES", 1)

    call_count = {"n": 0}

    def fake_chat(system_prompt, user_prompt, max_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Batch call: model merged lines 2 and 3 -> only 2 lines returned.
            return "1. unu\n2. doi si trei"
        # Per-cue fallback calls.
        return f"fallback-{call_count['n']}"

    monkeypatch.setattr(translate, "_chat", fake_chat)

    texts = ["one", "two", "three"]
    result = translate.translate_text(texts, "ro")

    assert len(result) == len(texts)
    assert result == ["fallback-2", "fallback-3", "fallback-4"]


def test_multiple_batches_use_context(monkeypatch):
    _patch_load(monkeypatch)
    monkeypatch.setattr(translate, "BATCH_SIZE", 2)
    monkeypatch.setattr(translate, "CONTEXT_CUES", 2)

    seen_prompts = []

    def fake_chat(system_prompt, user_prompt, max_tokens):
        seen_prompts.append(user_prompt)
        if "Context" in user_prompt:
            return "1. trei\n2. patru"
        return "1. unu\n2. doi"

    monkeypatch.setattr(translate, "_chat", fake_chat)

    texts = ["one", "two", "three", "four"]
    result = translate.translate_text(texts, "ro")

    assert len(result) == 4
    assert "Context" in seen_prompts[1]
    assert "one" in seen_prompts[1] and "two" in seen_prompts[1]
