import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import translate


class FakeSp:
    """Minimal sentencepiece stand-in: pieces are whitespace-split tokens."""

    def encode(self, text, out_type=str):
        return text.split()

    def decode(self, pieces):
        # Drop the leading "<2xx>" target token to mimic a real translation.
        return " ".join(p for p in pieces if not p.startswith("<2"))


class _Hyp:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class FakeTranslator:
    def __init__(self):
        self.seen = []

    def translate_batch(self, inputs, **kwargs):
        self.seen = inputs
        # Echo the input tokens back as the "translation".
        return [_Hyp(tokens) for tokens in inputs]


def _patch(monkeypatch):
    fake_tr = FakeTranslator()
    monkeypatch.setattr(translate, "_load", lambda: None)
    monkeypatch.setattr(translate, "_sp", FakeSp())
    monkeypatch.setattr(translate, "_translator", fake_tr)
    return fake_tr


def test_preserves_count_and_applies_target_token(monkeypatch):
    fake_tr = _patch(monkeypatch)

    texts = ["one", "two two", "three"]
    result = translate.translate_text(texts, "ro")

    assert len(result) == len(texts)
    # Every input sent to the model is prefixed with the target-language token.
    assert all(tokens[0] == "<2ro>" for tokens in fake_tr.seen)


def test_empty_strings_pass_through(monkeypatch):
    fake_tr = _patch(monkeypatch)

    result = translate.translate_text(["", "   ", "hello"], "ro")

    assert result[0] == "" and result[1] == ""
    assert result[2] != ""
    # Only the non-blank cue is sent to the model.
    assert len(fake_tr.seen) == 1


def test_unsupported_lang_falls_back_to_ro(monkeypatch):
    fake_tr = _patch(monkeypatch)

    translate.translate_text(["something"], "zz")

    assert fake_tr.seen[0][0] == "<2ro>"


def test_supported_lang_uses_its_token(monkeypatch):
    fake_tr = _patch(monkeypatch)

    translate.translate_text(["something"], "fr")

    assert fake_tr.seen[0][0] == "<2fr>"
