#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PII Scanner — финальное решение хакатона
========================================
Обходит датасет вида:
  <root>/
    data/    *.csv, *.parquet
    text/    *.txt
    images/  *.tif, *.tiff

Для каждого файла:
  1. Извлекает текст (с OCR для изображений).
  2. Ищет признаки ПДн: ФИО, email, телефон, СНИЛС, ИНН,
     паспорт РФ, MRZ, платёжные реквизиты, биометрия,
     специальные категории.
  3. Файлы с ПДн записывает в result.csv (size, time, name).
  4. Дополнительно сохраняет scan_debug.csv и scan_all.log.

Запуск:
  python scan_all.py <path_to_root> [<tesseract_exe>]
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# CONFIG
# ============================================================

DEFAULT_ROOT = r"C:\PyProject\XaXaTon\ПДнDataset\share"
DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUTPUT_CSV = "result.csv"
FALLBACK_OUTPUT_CSV = "result_new.csv"
DEBUG_CSV = "scan_debug.csv"
LOG_FILE = "scan_all.log"
CONVERTED_DIRNAME = "converted_images"

ALLOWED_TOP_DIRS = {"data", "text", "images"}

MAX_TEXT_CHARS = 1_500_000
MAX_ROWS_READ = 5_000
MAX_DEBUG_MATCHES = 5

# ============================================================
# STOP-СЛОВА (топонимы и организации — снижают ложные ФИО)
# ============================================================

TOPONYM_WORDS = {
    'новгород', 'москва', 'санкт', 'петербург', 'ростов', 'краснодар',
    'екатеринбург', 'новосибирск', 'казань', 'уфа', 'омск', 'самара',
    'посад', 'мартан', 'челны', 'набережные', 'александров', 'сергиев',
    'разина', 'степана', 'ставрополь', 'воронеж', 'волгоград', 'пермь',
    'красноярск', 'саратов', 'тюмень', 'тольятти', 'ижевск', 'барнаул',
    'хабаровск', 'владивосток', 'ярославль', 'иркутск', 'махачкала',
    'томск', 'оренбург', 'кемерово', 'рязань', 'пенза', 'липецк',
    'лесной', 'рубеж', 'учебный', 'банк', 'южное', 'бутово',
    'тихая', 'слобода', 'летняя', 'долина', 'речной', 'квартал',
    'звёздная', 'роща', 'школьная', 'гавань', 'академика',
    'синтек', 'вектор', 'тех', 'лабс', 'улица', 'проспект', 'площадь',
    'компания', 'общество', 'федерация', 'российская', 'министерство',
    'департамент',
}

# ============================================================
# ПАТТЕРНЫ ПДн
# ============================================================

# Обычные
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
FIO_RE   = re.compile(r"\b[А-ЯЁ][а-яё]{2,20}\s+[А-ЯЁ][а-яё]{2,20}(?:\s+[А-ЯЁ][а-яё]{2,20})?\b")
DOB_RE   = re.compile(r"\b\d{2}[./]\d{2}[./]\d{4}\b")
INDEX_RE = re.compile(r"\b\d{6}\b")

# Государственные идентификаторы
SNILS_RE    = re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b")
INN12_RE    = re.compile(r"(?<!\d)\d{12}(?!\d)")
INN10_RE    = re.compile(r"(?<!\d)\d{10}(?!\d)")
PASSPORT_RE = re.compile(r"\b\d{4}\s\d{6}\b")
MRZ_RE      = re.compile(r"[A-Z0-9<]{20,}")
DL_RE       = re.compile(r"(?<!\d)\d{10,12}(?!\d)")

# Платёжные
CARD_RE = re.compile(r"(?:(?:\d[ \-]*?){13,19})")
CVV_RE  = re.compile(r"\b(CVV|CVC|CVV2)\b", re.IGNORECASE)
RS_RE   = re.compile(r"(?i)(?:р/с|расч[её]тн(?:ый)?\s+сч[её]т)[^\d]*(\d{20})")
BIK_RE  = re.compile(r"(?i)бик[^\d]*(\d{9})")

# Ключевые слова: биометрия
BIOMETRIC_KEYS = [
    'биометр', 'отпечат', 'радуж', 'ирис', 'лицев', 'селфи',
    'faceid', 'fingerprint', 'voiceprint', 'голосов', 'геометрия лица',
]

# Ключевые слова: специальные категории (здоровье, убеждения)
SPECIAL_KEYS = [
    'диагноз', 'анамнез', 'инвалид', 'здоровь', 'медицин', 'психиатр',
    'вич', 'религ', 'вероисповед', 'политическ', 'партия', 'интим', 'сексуаль',
]

# Ключевые слова документов (для score_document_text)
DOC_KEYWORDS = [
    # Universal
    "passport", "passaporte", "personalausweis", "führerschein", "fuhrerschein",
    "identity card", "identification card", "driver", "licence", "license",
    "date of birth", "date of expiry", "issuing country", "authority",
    "nationality", "surname", "given names", "passport no", "personal no",
    # German
    "bundesrepublik deutschland", "geburtsort", "geburtstag",
    "vorname", "nachname", "gültig bis", "gultig bis", "muster",
    # Spanish / Portuguese / Latin American
    "republica federativa do brasil", "república federativa do brasil",
    "republica de chile", "república de chile",
    "cédula", "cedula", "identidad", "nacimiento", "apellidos", "nombres",
    "nacionalidad", "fecha de", "run ", "rut ", "pasaporte",
    # Czech / Slovak
    "česká republika", "ceska republika", "občanský průkaz", "obcansky prukaz",
    "datum narození", "místo narození", "platnost do",
    # Romanian
    "romania", "românia", "permis de conducere",
    # Albanian
    "republika e shqipërisë", "republic of albania", "lëtërnjoftim",
    # Arabic
    "جواز السفر", "الجنسية", "الاسم",
    # Chinese
    "中华人民共和国", "居民身份证", "签发机关", "有效期限",
    # Russian
    "паспорт", "дата рождения", "место рождения", "гражданство",
]

DATE_RE = re.compile(
    r"\b(?:\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{1,2}\s+[A-Z]{3,9}\s+\d{2,4})\b",
    re.IGNORECASE,
)

# ============================================================
# ВАЛИДАТОРЫ
# ============================================================

def luhn_check(number: str) -> bool:
    digits = [int(d) for d in re.sub(r'\D', '', number)]
    if not (13 <= len(digits) <= 19):
        return False
    s, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0

def snils_valid(snils: str) -> bool:
    nums = re.sub(r'\D', '', snils)
    if len(nums) != 11:
        return False
    base  = [int(x) for x in nums[:9]]
    check = int(nums[9:])
    s = sum((9 - i) * d for i, d in enumerate(base))
    c = s if s < 100 else (0 if s in (100, 101) else (s % 101 % 100))
    return c == check

def inn_valid(inn: str) -> bool:
    nums = re.sub(r'\D', '', inn)
    if len(nums) == 10:
        w = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        c = sum(int(nums[i]) * w[i] for i in range(9)) % 11 % 10
        return c == int(nums[9])
    elif len(nums) == 12:
        w1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
        w2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
        c1 = sum(int(nums[i]) * w1[i] for i in range(11)) % 11 % 10
        c2 = sum(int(nums[i]) * w2[i] for i in range(11)) % 11 % 10
        return c1 == int(nums[10]) and c2 == int(nums[11])
    return False

def has_context(text: str, idx: int, window: int, *keywords: str) -> bool:
    chunk = text[max(0, idx - window): idx + window]
    return any(k in chunk for k in keywords)

# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class ScanDecision:
    path: Path
    top_dir: str
    source_kind: str
    is_pdn: bool
    categories: Dict[str, int] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    matches: Dict[str, List[str]] = field(default_factory=dict)
    doc_score: float = 0.0
    elapsed_sec: float = 0.0
    ocr_source: str = ""

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

def log(msg: str, fh=None) -> None:
    print(msg)
    if fh:
        fh.write(msg + "\n")
        fh.flush()

# ============================================================
# УТИЛИТЫ
# ============================================================

def fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%b %d %H:%M").lower().replace(" 0", " ")

def normalize_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def top_dir_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else ""

def is_toponym_like(s: str) -> bool:
    low = s.lower()
    return any(word in low for word in TOPONYM_WORDS)

def clean_fio_matches(vals: List[str]) -> List[str]:
    return [v for v in vals if not is_toponym_like(v)]

# ============================================================
# ИТЕРАЦИЯ ФАЙЛОВ
# ============================================================

def iter_candidate_files(root: Path):
    data_files, text_files, image_files = [], [], []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            top = top_dir_of(path, root)
        except Exception:
            continue
        if top not in ALLOWED_TOP_DIRS:
            continue
        ext = path.suffix.lower()
        if top == "data"   and ext in {".csv", ".parquet"}:
            data_files.append(path)
        elif top == "text" and ext == ".txt":
            text_files.append(path)
        elif top == "images" and ext in {".tif", ".tiff"}:
            image_files.append(path)

    yield from sorted(data_files,  key=lambda p: p.name.lower())
    yield from sorted(text_files,  key=lambda p: p.name.lower())
    yield from sorted(image_files, key=lambda p: p.name.lower())

# ============================================================
# ЭКСТРАКТОРЫ ТЕКСТА
# ============================================================

def read_text_fallback(path: Path) -> str:
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="replace")[:MAX_TEXT_CHARS]
        except Exception:
            pass
    return ""

def extract_from_csv(path: Path) -> str:
    lines = []
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                for i, line in enumerate(f):
                    if i >= MAX_ROWS_READ:
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines)
        except Exception:
            pass
    return ""

def extract_from_parquet(path: Path) -> str:
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        if len(df) > 3000:
            df = df.head(3000)
        return df.astype(str).to_csv(index=False, sep=" ")[:MAX_TEXT_CHARS]
    except Exception:
        return ""

def ensure_converted_dir(root: Path) -> Path:
    out = root / CONVERTED_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out

def convert_tif_to_jpg(src: Path, converted_dir: Path) -> Optional[Path]:
    try:
        from PIL import Image, ImageOps
        dst = converted_dir / f"{src.stem}.jpg"
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst
        img = Image.open(src)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if img.mode == "L":
            img = ImageOps.autocontrast(img)
        w, h = img.size
        if max(w, h) > 2500:
            scale = 2500 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        if img.mode == "L":
            img = img.convert("RGB")
        img.save(dst, format="JPEG", quality=90, optimize=True)
        return dst
    except Exception:
        return None

def extract_from_image(path: Path, converted_dir: Path, tesseract_cmd: Optional[str]) -> Tuple[str, str]:
    """Возвращает (text, ocr_source)."""
    try:
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        ext = path.suffix.lower()
        ocr_path = path
        ocr_source = "original_image"

        if ext in {".tif", ".tiff"}:
            jpg = convert_tif_to_jpg(path, converted_dir)
            if jpg and jpg.exists():
                ocr_path, ocr_source = jpg, "converted_jpg"
            else:
                ocr_source = "original_tif"

        data = np.fromfile(str(ocr_path), dtype=np.uint8)
        if data.size == 0:
            return "", "failed"

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            try:
                pil = Image.open(ocr_path)
                if pil.mode not in ("RGB", "L"):
                    pil = pil.convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                return "", "failed"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Четыре варианта предобработки → максимальный охват символов
        variants = []

        # v1: equalizeHist + 2x upscale
        v1 = cv2.equalizeHist(gray)
        v1 = cv2.resize(v1, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(v1)

        # v2: CLAHE + adaptive threshold (хорошо для бликов/ламинации)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v2 = clahe.apply(gray)
        v2 = cv2.GaussianBlur(v2, (3, 3), 0)
        v2 = cv2.adaptiveThreshold(
            v2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        v2 = cv2.resize(v2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(v2)

        # v3: sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        v3 = cv2.filter2D(gray, -1, kernel)
        v3 = cv2.resize(v3, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(v3)

        # v4: Otsu binarization (хорошо для чётких карточек и MRZ)
        _, v4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v4 = cv2.resize(v4, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(v4)

        texts = []
        for v in variants:
            pil = Image.fromarray(v)
            for psm in (6, 11, 4):
                for lang in ("eng+rus", "chi_sim+chi_tra+eng", "eng"):
                    try:
                        txt = pytesseract.image_to_string(
                            pil, lang=lang, config=f"--oem 3 --psm {psm}"
                        )
                        txt = normalize_text(txt)
                        if txt:
                            texts.append(txt)
                            break  # use first lang that works
                    except Exception:
                        pass

        return "\n".join(texts)[:MAX_TEXT_CHARS], ocr_source

    except Exception:
        return "", "failed"

# ============================================================
# ДЕТЕКЦИЯ ПДн
# ============================================================

def detect_categories(text: str) -> Dict[str, int]:
    """Возвращает словарь категория → количество находок."""
    t   = text if isinstance(text, str) else ""
    low = t.lower()
    cats = {
        "обычные": 0,
        "государственные": 0,
        "платёжные": 0,
        "биометрические": 0,
        "специальные": 0,
    }

    # --- Обычные ---
    cats["обычные"] += len(EMAIL_RE.findall(t))
    cats["обычные"] += len(PHONE_RE.findall(t))
    cats["обычные"] += min(10, len(FIO_RE.findall(t)))          # cap, чтобы не раздувать
    for m in DOB_RE.finditer(t):
        if has_context(low, m.start(), 40, "дата рождения", "родил"):
            cats["обычные"] += 1
    for m in INDEX_RE.finditer(t):
        if has_context(low, m.start(), 40, "ул", "улица", "просп", "пер",
                       "дом", "квартира", "город", "г."):
            cats["обычные"] += 1

    # --- Государственные ---
    for m in SNILS_RE.finditer(t):
        if snils_valid(m.group(0)):
            cats["государственные"] += 1
    for m in INN12_RE.finditer(t):
        if inn_valid(m.group(0)):
            cats["государственные"] += 1
    for m in INN10_RE.finditer(t):
        if inn_valid(m.group(0)):
            cats["государственные"] += 1
    for m in PASSPORT_RE.finditer(t):
        if has_context(low, m.start(), 60,
                       "паспорт", "серия", "номер", "код подразделения"):
            cats["государственные"] += 1
    for m in DL_RE.finditer(t):
        if has_context(low, m.start(), 30, "водител", "удостовер"):
            cats["государственные"] += 1
    if MRZ_RE.search(t):
        cats["государственные"] += 1

    # --- Платёжные ---
    for m in CARD_RE.finditer(t):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_check(digits):
            if has_context(low, m.start(), 40,
                           "visa", "mastercard", "карта", "cvv", "cvc",
                           "номер карты"):
                cats["платёжные"] += 1
    cats["платёжные"] += len(RS_RE.findall(t))
    cats["платёжные"] += len(BIK_RE.findall(t))
    if CVV_RE.search(t):
        cats["платёжные"] += 1

    # --- Биометрические ---
    if any(k in low for k in BIOMETRIC_KEYS):
        cats["биометрические"] += 1

    # --- Специальные ---
    if any(k in low for k in SPECIAL_KEYS):
        cats["специальные"] += 1

    return cats

def estimate_uz(cats: Dict[str, int]) -> str:
    """Эвристика уровня защищённости (152-ФЗ)."""
    total    = sum(cats.values())
    distinct = sum(1 for v in cats.values() if v > 0)
    has_s = cats["специальные"]    > 0
    has_b = cats["биометрические"] > 0
    has_p = cats["платёжные"]      > 0
    has_g = cats["государственные"]> 0
    has_c = cats["обычные"]        > 0

    if has_s or has_b:
        return "УЗ-1" if (total >= 5 or distinct >= 2) else "УЗ-2"
    if has_p or has_g:
        return "УЗ-2" if (total >= 5 or distinct >= 2) else "УЗ-3"
    if has_c:
        return "УЗ-3" if (total >= 5 or distinct >= 2) else "УЗ-4"
    return "нет признаков"

def extract_matches_for_debug(text: str) -> Dict[str, List[str]]:
    """Извлекает примеры найденных значений (для debug-CSV)."""
    found: Dict[str, List[str]] = {}
    seen = set()
    simple_patterns = {
        "EMAIL":       EMAIL_RE,
        "PHONE":       PHONE_RE,
        "SNILS":       SNILS_RE,
        "INN":         INN12_RE,
        "PASSPORT_RU": PASSPORT_RE,
        "FIO_RU":      FIO_RE,
    }
    for cat, pat in simple_patterns.items():
        for m in pat.findall(text):
            m = m.strip()
            if not m or (cat, m) in seen:
                continue
            seen.add((cat, m))
            found.setdefault(cat, []).append(m)

    if "FIO_RU" in found:
        found["FIO_RU"] = clean_fio_matches(found["FIO_RU"])
        if not found["FIO_RU"]:
            del found["FIO_RU"]

    return found

# ============================================================
# DOCUMENT SCORING (для изображений/документов)
# ============================================================

def score_document_text(text: str) -> Tuple[float, List[str]]:
    low = text.lower()
    score, reasons = 0.0, []

    kw_hits = [kw for kw in DOC_KEYWORDS if kw.lower() in low]
    if kw_hits:
        score += min(3.0, 0.6 * len(set(kw_hits)))
        reasons.append(f"kw={len(set(kw_hits))}")

    mrz_hits = MRZ_RE.findall(text)
    if mrz_hits:
        score += min(2.4, 0.8 * len(mrz_hits))
        reasons.append("mrz")

    date_hits = DATE_RE.findall(text)
    if len(date_hits) >= 2:
        score += 0.8
        reasons.append(f"dates={len(date_hits)}")

    combo_terms = ["surname", "given", "nationality", "birth", "expiry"]
    combo_count = sum(1 for t in combo_terms if t in low)
    if combo_count >= 3:
        score += 1.0
        reasons.append("id_combo")

    return score, reasons

# ============================================================
# ПРАВИЛА РЕШЕНИЯ ПО ТИПАМ ФАЙЛОВ
# ============================================================

def decide(top_dir: str, path: Path, cats: Dict[str, int],
           text: str) -> Tuple[bool, List[str], float]:
    doc_score, doc_reasons = score_document_text(text)
    reasons: List[str] = []

    has_strong_id = (
        cats.get("государственные", 0) > 0 or
        cats.get("платёжные", 0) > 0 or
        cats.get("биометрические", 0) > 0 or
        cats.get("специальные", 0) > 0
    )
    has_email_or_phone = (
        EMAIL_RE.search(text) is not None or
        PHONE_RE.search(text) is not None
    )
    fio_count = len(clean_fio_matches(FIO_RE.findall(text)))

    if top_dir == "data":
        if has_strong_id or has_email_or_phone:
            reasons.append("strong")
            return True, reasons + doc_reasons, doc_score
        if path.name.lower() in {"customers.csv", "logistics.csv"} and fio_count >= 10:
            reasons.append(f"fio_count={fio_count}")
            return True, reasons + doc_reasons, doc_score

    elif top_dir == "text":
        if has_strong_id or has_email_or_phone:
            reasons.append("strong")
            return True, reasons + doc_reasons, doc_score
        if fio_count >= 2:
            reasons.append(f"fio_count={fio_count}")
            return True, reasons + doc_reasons, doc_score

    elif top_dir == "images":
        if has_strong_id or has_email_or_phone:
            reasons.append("strong")
            return True, reasons + doc_reasons, doc_score
        if doc_score >= 2.0:                          # было 3.0
            reasons.append("doc_score>=2")
            return True, reasons + doc_reasons, doc_score
        if doc_score >= 1.0 and fio_count >= 1:       # было 2.0
            reasons.append("doc_score+fio")
            return True, reasons + doc_reasons, doc_score
        # Fallback: много дат в тексте — скорее всего структурированный документ
        date_count = len(DATE_RE.findall(text))
        if date_count >= 2 and doc_score >= 0.8:     # КА10_04 и аналоги
            reasons.append(f"dates_score={date_count}")
            return True, reasons + doc_reasons, doc_score
        if date_count >= 3:                           # строгий date-only fallback
            reasons.append(f"dates_fallback={date_count}")
            return True, reasons + doc_reasons, doc_score

    return False, doc_reasons, doc_score

# ============================================================
# СКАНИРОВАНИЕ ОДНОГО ФАЙЛА
# ============================================================

def scan_one(path: Path, root: Path, converted_dir: Path,
             tesseract_cmd: Optional[str]) -> Optional[ScanDecision]:
    top = top_dir_of(path, root)
    ext = path.suffix.lower()
    started = time.time()
    text, kind, ocr_source = "", "", ""

    if top == "data":
        if ext == ".csv":
            text, kind = extract_from_csv(path), "csv"
        elif ext == ".parquet":
            text, kind = extract_from_parquet(path), "parquet"
        else:
            return None

    elif top == "text":
        if ext == ".txt":
            text, kind = read_text_fallback(path), "txt"
        else:
            return None

    elif top == "images":
        if ext in {".tif", ".tiff"}:
            text, ocr_source = extract_from_image(path, converted_dir, tesseract_cmd)
            kind = "image"
        else:
            return None
    else:
        return None

    text    = normalize_text(text)
    elapsed = time.time() - started

    if not text:
        return ScanDecision(
            path=path, top_dir=top, source_kind=kind, is_pdn=False,
            reasons=["empty"], elapsed_sec=elapsed, ocr_source=ocr_source,
        )

    cats    = detect_categories(text)
    matches = extract_matches_for_debug(text)
    is_pdn, reasons, doc_score = decide(top, path, cats, text)

    return ScanDecision(
        path=path, top_dir=top, source_kind=kind,
        is_pdn=is_pdn, categories=cats, reasons=reasons,
        matches=matches, doc_score=doc_score,
        elapsed_sec=elapsed, ocr_source=ocr_source,
    )

# ============================================================
# ЗАПИСЬ РЕЗУЛЬТАТОВ
# ============================================================

def write_result(rows: List[Tuple], output_path: Path) -> Path:
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["size", "time", "name"])
            writer.writerows(rows)
        return output_path
    except PermissionError:
        fallback = output_path.with_name(FALLBACK_OUTPUT_CSV)
        with open(fallback, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["size", "time", "name"])
            writer.writerows(rows)
        return fallback

def write_debug(decisions: List[ScanDecision], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "top_dir", "kind", "is_pdn", "uz",
            "doc_score", "elapsed_sec", "ocr_source", "reasons", "matches",
        ])
        for d in decisions:
            uz = estimate_uz(d.categories) if d.categories else "—"
            writer.writerow([
                d.path.name,
                d.top_dir,
                d.source_kind,
                int(d.is_pdn),
                uz,
                f"{d.doc_score:.2f}",
                f"{d.elapsed_sec:.2f}",
                d.ocr_source,
                " | ".join(d.reasons),
                json.dumps(
                    {k: v[:MAX_DEBUG_MATCHES] for k, v in d.matches.items()},
                    ensure_ascii=False,
                ),
            ])

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    root          = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    tesseract_cmd = sys.argv[2]       if len(sys.argv) > 2 else DEFAULT_TESSERACT

    if not root.exists():
        print(f"ERROR: root not found: {root}")
        sys.exit(1)

    converted_dir = ensure_converted_dir(root)
    files         = list(iter_candidate_files(root))
    total         = len(files)

    decisions: List[ScanDecision] = []
    rows:      List[Tuple]        = []
    seen_names: set               = set()

    log_path    = root / LOG_FILE
    started_all = time.time()

    with open(log_path, "w", encoding="utf-8") as lf:
        log(f"ROOT          : {root}", lf)
        log(f"TESSERACT     : {tesseract_cmd}", lf)
        log(f"CONVERTED_DIR : {converted_dir}", lf)
        log(f"TOTAL FILES   : {total}", lf)
        log("", lf)

        for idx, path in enumerate(files, 1):
            top = top_dir_of(path, root)
            log(f"[{idx}/{total}] {top}/{path.name}", lf)

            d = scan_one(path, root, converted_dir, tesseract_cmd)
            if d is None:
                log("    SKIP (unsupported)", lf)
                continue

            decisions.append(d)
            uz          = estimate_uz(d.categories)
            match_counts = {k: len(v) for k, v in d.matches.items()}

            tag = "PDN " if d.is_pdn else "CLEAN"
            log(
                f"    {tag} kind={d.source_kind} uz={uz} "
                f"score={d.doc_score:.2f} t={d.elapsed_sec:.2f}s "
                f"ocr={d.ocr_source or '-'} "
                f"reasons={' | '.join(d.reasons[:5])} "
                f"matches={json.dumps(match_counts, ensure_ascii=False)}",
                lf,
            )

            if d.is_pdn and path.name not in seen_names:
                seen_names.add(path.name)
                st = path.stat()
                rows.append((st.st_size, fmt_mtime(st.st_mtime), path.name))

        rows.sort(key=lambda x: x[2].lower())

        out = write_result(rows, root / OUTPUT_CSV)
        write_debug(decisions, root / DEBUG_CSV)

        elapsed_all = time.time() - started_all
        log("", lf)
        log(f"result  → {out}", lf)
        log(f"debug   → {root / DEBUG_CSV}", lf)
        log(f"log     → {log_path}", lf)
        log(f"PDN files found : {len(rows)}", lf)
        log(f"Total time      : {elapsed_all:.2f}s", lf)

if __name__ == "__main__":
    main()