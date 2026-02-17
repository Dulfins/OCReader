import ctranslate2
from pathlib import Path
import sentencepiece as spm
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR  / "model"
SPM_PATH = BASE_DIR  / "spm.ja.nopretok.model"

@lru_cache(maxsize=1)
def load_translator():
    print("Loading CTranslate2 model...")
    translator = ctranslate2.Translator(
        str(MODEL_DIR),
        device="cpu"
    )
    print("Model loaded successfully")
    return translator

@lru_cache(maxsize=1)
def load_sp():
    """Load SentencePiece once (cached)."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_PATH))
    return sp

def translate_ja_to_en(sentences):
    """
    Translate a list of Japanese sentences to English.
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    sp = load_sp()
    translator = load_translator()

    tokenized = [sp.encode(s, out_type=str) for s in sentences]
    results = translator.translate_batch(tokenized)

    translations = []
    for r in results:
        tokens = r.hypotheses[0]
        text = sp.decode(tokens)

        # Cleaning
        text = " ".join(text.split("▁")).strip()

        translations.append(text)

    return translations