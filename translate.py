"""Translation via Madlad-400 (CTranslate2).

Madlad-400 is a dedicated multilingual NMT model. Compared to a general LLM it is
far faster on CPU and won't fabricate words, and it needs no source-language
detection -- you only prefix the source text with a target-language token
(`<2ro>` for Romanian). One translated cue is produced per input cue, so subtitle
timestamps always stay aligned.

The model must be a CTranslate2 build (a `model.bin` + a SentencePiece model),
e.g. `santhosh/madlad400-3b-ct2`. The plain HuggingFace/transformers repo
(`jbochi/madlad400-3b-mt`, safetensors) is NOT loadable by CTranslate2.
"""

import os
import threading
from pathlib import Path

MODEL_DIR = Path(os.environ.get("TRANSLATE_MODEL_DIR", str(Path(__file__).parent / "models")))
# Sub-directory holding the CTranslate2 model + SentencePiece model.
MODEL_SUBDIR = os.environ.get("TRANSLATE_MODEL_SUBDIR", "madlad-ct2")
MODEL_PATH = Path(os.environ.get("TRANSLATE_MODEL_PATH", str(MODEL_DIR / MODEL_SUBDIR)))
MODEL_REPO = os.environ.get("TRANSLATE_MODEL_REPO", "santhosh/madlad400-3b-ct2")

# CT2 quantizes to this on load, so the weights stay small in RAM (~3-4 GB).
COMPUTE_TYPE = os.environ.get("TRANSLATE_COMPUTE_TYPE", "int8")
THREADS = int(os.environ.get("TRANSLATE_THREADS", str(os.cpu_count() or 4)))
MAX_BATCH_SIZE = int(os.environ.get("TRANSLATE_MAX_BATCH_SIZE", "1024"))
BEAM_SIZE = int(os.environ.get("TRANSLATE_BEAM_SIZE", "1"))
# Madlad-CT2 anti-repetition knobs (tunable if you see looping/repeats).
REPETITION_PENALTY = float(os.environ.get("TRANSLATE_REPETITION_PENALTY", "1.1"))
NO_REPEAT_NGRAM = int(os.environ.get("TRANSLATE_NO_REPEAT_NGRAM", "0"))

# Target languages offered in the UI; all are valid Madlad `<2xx>` codes.
SUPPORTED_LANGS = {"ro", "en", "fr", "de", "es", "it", "pt", "nl", "ru"}
DEFAULT_LANG = "ro"

_SPM_CANDIDATES = ("sentencepiece.model", "spiece.model")

_translator = None
_sp = None
_lock = threading.Lock()


def _find_spm_file() -> Path:
    override = os.environ.get("TRANSLATE_SPM_FILE")
    if override:
        return MODEL_PATH / override
    for name in _SPM_CANDIDATES:
        if (MODEL_PATH / name).exists():
            return MODEL_PATH / name
    for pattern in ("*.model", "*.spm"):
        matches = sorted(MODEL_PATH.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No SentencePiece model file found in {MODEL_PATH}")


def _load():
    global _translator, _sp
    if _translator is not None:
        return
    import ctranslate2
    import sentencepiece as spm

    if not (MODEL_PATH / "model.bin").exists():
        raise FileNotFoundError(
            f"CTranslate2 model not found at {MODEL_PATH} (no model.bin). Download a CT2 "
            f"build with: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='{MODEL_REPO}', local_dir='{MODEL_PATH}')\""
        )
    _translator = ctranslate2.Translator(
        str(MODEL_PATH), device="cpu", compute_type=COMPUTE_TYPE, intra_threads=THREADS
    )
    _sp = spm.SentencePieceProcessor()
    _sp.load(str(_find_spm_file()))


def translate_text(texts: list[str], target_lang: str = "ro") -> list[str]:
    if not texts:
        return []

    _load()
    lang = target_lang if target_lang in SUPPORTED_LANGS else DEFAULT_LANG

    # Empty/blank cues translate to "" without bothering the model.
    indices = [i for i, t in enumerate(texts) if t and t.strip()]
    results = [""] * len(texts)
    if not indices:
        return results

    inputs = [_sp.encode(f"<2{lang}> {texts[i].strip()}", out_type=str) for i in indices]
    with _lock:
        outputs = _translator.translate_batch(
            inputs,
            batch_type="tokens",
            max_batch_size=MAX_BATCH_SIZE,
            beam_size=BEAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )
    for i, out in zip(indices, outputs):
        results[i] = _sp.decode(out.hypotheses[0]).strip()
    return results
