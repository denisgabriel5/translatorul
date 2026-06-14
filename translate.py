import os
import re
import threading
from pathlib import Path

MODEL_DIR = Path(os.environ.get("TRANSLATE_MODEL_DIR", str(Path(__file__).parent / "models")))
MODEL_FILE = os.environ.get("TRANSLATE_MODEL_FILE", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
MODEL_PATH = Path(os.environ.get("TRANSLATE_MODEL_PATH", str(MODEL_DIR / MODEL_FILE)))

N_CTX = int(os.environ.get("TRANSLATE_N_CTX", "8192"))
N_THREADS = int(os.environ.get("TRANSLATE_N_THREADS", str(os.cpu_count() or 4)))

# How many cues to translate per LLM call, and how many preceding source
# cues to show as read-only context so the model can keep sentences that
# span cue boundaries coherent.
BATCH_SIZE = int(os.environ.get("TRANSLATE_BATCH_SIZE", "12"))
CONTEXT_CUES = int(os.environ.get("TRANSLATE_CONTEXT_CUES", "3"))

_llm = None
_lock = threading.Lock()

LANG_NAMES = {
    "ro": "Romanian",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional subtitle translator. You will be given consecutive "
    "lines of subtitles from a single video, in their original order. Some lines "
    "may be sentence fragments that only make sense together with neighboring "
    "lines. Translate the numbered lines into {lang} naturally and accurately, "
    "using the context lines only to understand meaning and flow.\n\n"
    "Rules:\n"
    "- Output exactly one translated line per input line, in the same order.\n"
    "- Prefix each output line with its number and a period, e.g. '3. ...'.\n"
    "- Do not merge, split, skip, or add lines.\n"
    "- Do not translate or repeat the context lines.\n"
    "- Return only the numbered translations, no extra commentary."
)

PER_CUE_SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional translator. Translate the given text accurately and "
    "naturally into {lang}. Return ONLY the translation, no explanations, no "
    "notes, no quotation marks."
)

_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)[\.\)\:]\s*(.*)$")


def _load():
    global _llm
    if _llm is not None:
        return
    from llama_cpp import Llama

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run: python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='lmstudio-community/Qwen2.5-7B-Instruct-GGUF', "
            f"filename='{MODEL_FILE}', local_dir='{MODEL_DIR}')\""
        )
    _llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )


def _lang_name(target_lang: str) -> str:
    return LANG_NAMES.get(target_lang, target_lang)


def _chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    with _lock:
        response = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
    return response["choices"][0]["message"]["content"].strip()


def _translate_one(text: str, lang: str) -> str:
    system_prompt = PER_CUE_SYSTEM_PROMPT_TEMPLATE.format(lang=lang)
    return _chat(system_prompt, f"{text}", max_tokens=256)


def _translate_batch(batch: list[str], context: list[str], lang: str) -> list[str] | None:
    """Translate a batch of cues with surrounding context.

    Returns the translated lines (same length as `batch`) in order, or
    None if the model's output couldn't be reliably aligned back to the
    input lines.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang=lang)

    parts = []
    if context:
        context_block = "\n".join(context)
        parts.append(f"Context (preceding lines, do not translate):\n{context_block}")
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(batch))
    parts.append(f"Translate these lines:\n{numbered}")
    user_prompt = "\n\n".join(parts)

    result = _chat(system_prompt, user_prompt, max_tokens=256 * len(batch))

    translations: dict[int, str] = {}
    for line in result.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1))
        translations[idx] = match.group(2).strip()

    if set(translations.keys()) != set(range(1, len(batch) + 1)):
        return None

    return [translations[i] for i in range(1, len(batch) + 1)]


def translate_text(texts: list[str], target_lang: str = "ro") -> list[str]:
    if not texts:
        return []

    _load()
    lang = _lang_name(target_lang)
    results: list[str] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        context = texts[max(0, start - CONTEXT_CUES):start]

        translated = _translate_batch(batch, context, lang)
        if translated is None:
            # Fall back to per-cue translation so timestamps never desync.
            translated = [_translate_one(text, lang) for text in batch]

        results.extend(translated)

    return results
