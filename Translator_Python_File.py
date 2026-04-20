from Dialect_Detector import detect_dialect
# Load the OPUS-MT Arabic→English Model

from transformers import MarianMTModel, MarianTokenizer
import torch

MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Dialect Normalization Layer (Key Innovation)
# Simple normalization dictionaries per dialect
DIALECT_NORMALIZATION = {
    "EGY": {
    # Negation
    "مش": "ليس",
    "مش ب": "لا",

    # Verbs
    "رايح": "ذاهب",
    "جاي": "قادم",
    "عايز": "أريد",
    "عاوزه": "أريده",

    # Phrases (important)
    "مش تعبان": "لست متعبا",
    "مش فاهم": "لا أفهم",

    # Question words
    "فين": "أين",
    "ليه": "لماذا"

    },
    "LEV": {
    # Negation
    "مو": "ليس",
    "مش": "ليس",

    # Intent / desire
    "بدّي": "أريد",
    "بدي": "أريد",

    # Verbs
    "راح": "ذهب",   # context‑dependent, but safer than leaving it
    "إجا": "جاء",

    # Location/question
    "وين": "أين",
    "ليش": "لماذا"

    },
    "GLF": {
    # Desire
    "أبغى": "أريد",
    "أبي": "أريد",

    # Negation
    "مو": "ليس",

    # Question
    "وين": "أين",
    "ليش": "لماذا"

    },
    "IRQ": {
        
    # Negation
    "ماكو": "لا يوجد",
    "مو": "ليس",

    # Verbs
    "أريد": "أريد",  # often already MSA
    "أروح": "أذهب",

    # Question
    "وين": "أين",
    "ليش": "لماذا"

    },
    "MGH" : {
        
    # Negation
    "ما": "لا",

    # Desire
    "نبغي": "نريد",
    "بغيت": "أردت",

    # Question
    "فين": "أين",
    "علاش": "لماذا"

    }

}

def normalize_dialect(text: str, dialect: str):
    """
    Replace dialectal words with MSA equivalents.
    Returns normalized text + list of applied rules.
    """
    applied_rules = []

    rules = DIALECT_NORMALIZATION.get(dialect, {})
    tokens = text.split()

    normalized_tokens = []
    for token in tokens:
        if token in rules:
            normalized_tokens.append(rules[token])
            applied_rules.append(f"{token} → {rules[token]}")
        else:
            normalized_tokens.append(token)

    normalized_text = " ".join(normalized_tokens)
    return normalized_text, applied_rules
# Translation Function (OPUS-MT Inference)

def translate_ar_to_en(text: str, max_length: int = 128):
    """
    Translate Arabic text to English using OPUS-MT.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        translated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(translated[0], skip_special_tokens=True)
# Explanation Layer (Ambiguity Awareness)
# This is post-processing, not modeling

AMBIGUOUS_WORDS = {
    "راح": ["went", "will go"],
    "كان": ["was", "used to be"],
    "طلع": ["went out", "turned out", "appeared"]
}

def get_ambiguity_notes(text: str):
    notes = []
    for word, meanings in AMBIGUOUS_WORDS.items():
        if word in text:
            notes.append({
                "word": word,
                "possible_meanings": meanings
            })
    return notes
# Main Pipeline (This Is What You Demo)
# Assumes You already have detect_dialect(text) implemented.


def dialect_aware_translate(text: str, detect_dialect):
    """
    Full pipeline:
    dialect detection → normalization → translation → explanations
    """
    dialect = detect_dialect(text)

    normalized_text, applied_rules = normalize_dialect(text, dialect)

    translation = translate_ar_to_en(normalized_text)

    ambiguity_notes = get_ambiguity_notes(text)

    return {
        "input_text": text,
        "detected_dialect": dialect,
        "normalized_text": normalized_text,
        "applied_normalization_rules": applied_rules,
        "translation": translation,
        "ambiguities": ambiguity_notes
    }
# Example Run (Perfect for Live Demo)

example = "19,حد يقدر يقول غير كدة"

result = dialect_aware_translate(example, detect_dialect)

for k, v in result.items():
    print(f"{k}: {v}")
