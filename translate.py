"""Translation via Madlad-400 (CTranslate2).

Madlad-400 is a dedicated multilingual NMT model. Compared to a general LLM it is
far faster on CPU and won't fabricate words, and it needs no source-language
detection -- you only prefix the source text with a target-language token
(`<2ro>` for Romanian). One translated cue is produced per input cue, so subtitle
timestamps always stay aligned.
"""

import os
import threading
from pathlib import Path

MODEL_DIR = Path(os.environ.get("TRANSLATE_MODEL_DIR", str(Path(__file__).parent / "models")))
# Sub-directory holding the CTranslate2 model + sentencepiece.model.
MODEL_SUBDIR = os.environ.get("TRANSLATE_MODEL_SUBDIR", "madlad")
MODEL_PATH = Path(os.environ.get("TRANSLATE_MODEL_PATH", str(MODEL_DIR / MODEL_SUBDIR)))
MODEL_REPO = os.environ.get("TRANSLATE_MODEL_REPO", "jbochi/madlad400-3b-mt")
SPM_FILE = os.environ.get("TRANSLATE_SPM_FILE", "sentencepiece.model")

# CT2 quantizes to this on load, so the float weights stay small in RAM.
COMPUTE_TYPE = os.environ.get("TRANSLATE_COMPUTE_TYPE", "int8")
THREADS = int(os.environ.get("TRANSLATE_THREADS", str(os.cpu_count() or 4)))
BATCH_SIZE = int(os.environ.get("TRANSLATE_BATCH_SIZE", "16"))
BEAM_SIZE = int(os.environ.get("TRANSLATE_BEAM_SIZE", "1"))

# Target languages offered in the UI; all are valid Madlad `<2xx>` codes.
SUPPORTED_LANGS = {"ro", "en", "fr", "de", "es", "it", "pt", "nl", "ru"}
DEFAULT_LANG = "ro"

_translator = None
_sp = None
_lock = threading.Lock()


def _load():
    global _translator, _sp
    if _translator is not None:
        return
    import ctranslate2
    import sentencepiece as spm

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Translation model not found at {MODEL_PATH}. Download it with: "
            f"python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='{MODEL_REPO}', local_dir='{MODEL_PATH}', "
            f"ignore_patterns=['*.safetensors'])\""
        )
    _translator = ctranslate2.Translator(
        str(MODEL_PATH), device="cpu", compute_type=COMPUTE_TYPE, intra_threads=THREADS
    )
    _sp = spm.SentencePieceProcessor()
    _sp.load(str(MODEL_PATH / SPM_FILE))


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
            inputs, max_batch_size=BATCH_SIZE, beam_size=BEAM_SIZE
        )
    for i, out in zip(indices, outputs):
        results[i] = _sp.decode(out.hypotheses[0]).strip()
    return results
