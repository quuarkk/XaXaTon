# XaXaTon

# ПДн-Сканер — поиск персональных данных в файловом хранилище

Решение задачи по обнаружению персональных данных (152-ФЗ) в корпоративном файловом хранилище.

## Описание решения

Скрипт `Reshenie.py` сканирует директорию с файлами и находит все файлы, содержащие персональные данные. Результат сохраняется в `result.csv` в формате `size,time,name`.

### Что обнаруживается

| Категория | Примеры |
|-----------|---------|
| ФИО | Русские и латинские имена, поля surname/given name в ID-картах |
| Email | любой@домен.ru |
| Телефоны | +7, 8, международные (+420, +49, ...) |
| Дата рождения | Русский, английский, чешский, немецкий языки |
| Паспорт РФ | Серия + номер, MRZ-строки |
| СНИЛС / ИНН | С контекстным словом или разделителями |
| Международные документы | Document no, ID card, Czech obcanský průkaz |
| Банковские карты | Проверка алгоритмом Луна, CVV, БИК, счета |
| Адреса | Российский формат (г. Москва, ул. Ленина, д. 1) |
| Биометрия | Упоминания отпечатков, сетчатки, распознавания лиц |
| Медданные | Диагнозы, МКБ-10, история болезни |

### Поддерживаемые форматы

**Текст / структура (фаза 1, CPU-параллельно):**
PDF, DOCX, DOC, RTF, TXT, MD, HTML, XML, CSV, TSV, JSON, JSONL, XLSX, XLS, Parquet, SQLite, ZIP, GZIP

**OCR — изображения (фаза 2, GPU):**
TIF, TIFF, JPG, JPEG, PNG, GIF, BMP

**OCR — видео (фаза 2, GPU):**
MP4, AVI, MOV (сэмплирование кадров каждые 2-5 сек)

### Архитектура

```
Датасет
   │
   ├─► Фаза 1: CPU ProcessPool (16 воркеров)
   │     PDF/DOCX/HTML/CSV/... → извлечение текста → regex ПДн
   │     PDF без текстового слоя → pending_ocr → фаза 2
   │
   └─► Фаза 2: OCR GPU (1 воркер, EasyOCR)
         Предобработка: TIF→JPEG (~500кб), PDF deflate-сжатие
         EasyOCR GPU → текст → regex ПДн
         Зависшие файлы → очередь повтора (таймаут ×3)

Checkpoint: results.jsonl (построчно, resume при падении)
Выход: result.csv + детальные отчёты CSV/JSON/Markdown
```

---

## Установка

### 1. Создать виртуальное окружение

```bash
python -m venv venv_pdn
# Windows:
venv_pdn\Scripts\activate
# Linux/Mac:
source venv_pdn/bin/activate
```

### 2. Установить зависимости

```bash
pip install pandas pdfplumber "pdfminer.six" PyPDF2 python-docx \
            beautifulsoup4 lxml openpyxl xlrd pyarrow chardet \
            docx2txt pytesseract Pillow olefile pymupdf \
            python-magic-bin opencv-python easyocr
```

### 3. Установить PyTorch с поддержкой CUDA (для GPU)

```bash
# CUDA 12.8 (RTX 40xx / 50xx):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 4. Установить Tesseract OCR (резервный движок)

**Windows:** https://github.com/UB-Mannheim/tesseract/wiki  
⚠️ При установке обязательно выбрать пакет языка **Russian**

**Linux:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-rus
```

---

## Запуск

### Основная команда

```bash
python Reshenie.py /path/to/dataset
```

### С GPU-ускорением (рекомендуется)

```bash
python Reshenie.py /path/to/dataset --ocr-engine easyocr --gpu-batch-size 16
```

### Продолжить прерванный скан

```bash
python Reshenie.py /path/to/dataset --resume
```

### Только текстовые файлы (без OCR изображений)

```bash
python Reshenie.py /path/to/dataset --no-ocr
```

### Все параметры

```
python Reshenie.py --help

Positional:
  directory                   Путь к директории с датасетом

Optional:
  --workers N                 CPU-воркеры для фазы 1 (по умолч: число ядер)
  --ocr-workers N             OCR-воркеры для фазы 2 (по умолч: 2)
  --ocr-engine ENGINE         auto | easyocr | tesseract | none
  --gpu-batch-size N          Батч EasyOCR (по умолч: 8; рекоменд: 16-32)
  --no-ocr                    Пропустить OCR
  --resume                    Продолжить с чекпоинта
  --out-dir DIR               Директория для отчётов (по умолч: ./pdn_reports)
  --result-csv FILE           Путь к result.csv (по умолч: ./result.csv)
  --submission-name-mode      basename | relative (по умолч: basename)
```

---

## Выходные файлы

```
result.csv                      ← Файл для сдачи (size, time, name)
pdn_reports/
    pdn_report_TIMESTAMP.csv    ← Детальный отчёт со всеми находками
    pdn_report_TIMESTAMP.json   ← То же в JSON
    pdn_report_TIMESTAMP.md     ← Markdown с примерами совпадений
    results.jsonl               ← Чекпоинт (для --resume)
```

### Формат result.csv

```csv
size,time,name
3068287,sep 26 18:31,CA01_01.tif
24983,sep 26 12:08,anketa.docx
```

- `size` — размер файла в байтах
- `time` — дата модификации (`mon dd HH:MM`, строчные, день без ведущего нуля)
- `name` — имя файла (basename)

---

## Производительность

Тестировалось на датасете ~3300 файлов (2.2 ГБ):

| Фаза | Движок | Скорость |
|------|--------|----------|
| Фаза 1 (текст) | 16 CPU воркеров | ~65 файл/с |
| Фаза 2 (OCR) | EasyOCR + RTX 5060 Ti | ~3-4 файл/с |
| Итого | | ~5-7 минут |

---

## Метрика

```
Score = (TP - 0.1 × FP) / (TP + FN)
```

Стратегия: при K=0.1 пропуск реального файла с ПДн (FN) в 10 раз дороже
ложного срабатывания (FP), поэтому пороги детекции настроены на высокий recall.
