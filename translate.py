import os
from pathlib import Path

from llama_cpp import Llama

MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_FILE

_llm = None

SYSTEM_PROMPT = (
    "You are a professional English-to-Romanian translator. "
    "Translate the given text accurately and naturally into Romanian. "
    "Return ONLY the translation, no explanations, no notes, no quotation marks."
)


def _load():
    global _llm
    if _llm is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run: python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='lmstudio-community/Qwen2.5-7B-Instruct-GGUF', "
            f"filename='{MODEL_FILE}', local_dir='{MODEL_DIR}')\""
        )
    _llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )


def translate_text(texts: list[str], target_lang: str = "ro") -> list[str]:
    _load()
    results = []
    for text in texts:
        response = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"English: {text}\nRomanian:"},
            ],
            max_tokens=256,
            temperature=0,
        )
        result = response["choices"][0]["message"]["content"].strip()
        results.append(result)
    return results
