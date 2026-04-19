#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organize_by_real_format_recursive.py — Рекурсивная реорганизация датасета.

Особенности:
  - Полностью рекурсивный обход всех подпапок
  - Определение реального формата по magic bytes
  - Сохранение плоской структуры (все PDF в pdf/, все DOC в office/ и т.д.)
  - Обнаружение подмен (HTML в .pdf, бинарники в .txt и т.п.)
  - Опциональное извлечение ZIP-архивов
  - Защита от коллизий имён (добавление суффиксов)
  - Сохранение отчёта о реорганизации

Запуск:
    python organize_by_real_format_recursive.py /path/to/dataset --mode move
    python organize_by_real_format_recursive.py /path/to/dataset --mode copy --output /new/location
    python organize_by_real_format_recursive.py /path/to/dataset --dry-run
"""

import argparse
import os
import shutil
import sys
import zipfile
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("organizer")

# ============================================================
# 1. MAGIC BYTES — сигнатуры форматов (расширенные)
# ============================================================
MAGIC_SIGNATURES: List[Tuple[bytes, int, str, str, List[str]]] = [
    # Документы
    (b"%PDF", 0, "pdf", "PDF", [".pdf"]),
    (b"PK\x03\x04", 0, "zip", "ZIP", [".zip", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub", ".jar", ".apk"]),
    (b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1", 0, "ole2", "OLE2 (DOC/XLS/PPT)", [".doc", ".xls", ".ppt"]),
    (b"{\\rtf", 0, "rtf", "RTF", [".rtf"]),
    
    # Изображения
    (b"\x89PNG\r\n\x1a\n", 0, "images", "PNG", [".png"]),
    (b"\xff\xd8\xff", 0, "images", "JPEG", [".jpg", ".jpeg"]),
    (b"GIF87a", 0, "images", "GIF", [".gif"]),
    (b"GIF89a", 0, "images", "GIF", [".gif"]),
    (b"BM", 0, "images", "BMP", [".bmp"]),
    (b"II\x2a\x00", 0, "images", "TIFF", [".tif", ".tiff"]),
    (b"MM\x00\x2a", 0, "images", "TIFF", [".tif", ".tiff"]),
    
    # Видео
    (b"\x00\x00\x00\x18ftypmp4", 4, "video", "MP4", [".mp4"]),
    (b"\x00\x00\x00\x1cftypisom", 4, "video", "MP4", [".mp4"]),
    (b"RIFF", 0, "video", "AVI", [".avi"]),
    (b"OggS", 0, "video", "OGG", [".ogv", ".ogg"]),
    
    # Архивы
    (b"\x1f\x8b", 0, "archives", "GZIP", [".gz", ".tgz"]),
    (b"7z\xbc\xaf'\x1c", 0, "archives", "7ZIP", [".7z"]),
    (b"Rar!\x1a\x07", 0, "archives", "RAR", [".rar"]),
    
    # Базы данных
    (b"SQLite format 3\x00", 0, "sqlite", "SQLite3", [".sqlite", ".db", ".sqlite3"]),
    
    # Исполняемые
    (b"\x7fELF", 0, "binary", "ELF", [".elf", ""]),
    (b"MZ", 0, "binary", "PE (EXE/DLL)", [".exe", ".dll"]),
    
    # Разметка
    (b"<?xml", 0, "xml", "XML", [".xml", ".xaml", ".svg"]),
    (b"<html", 0, "html", "HTML", [".html", ".htm"]),
    (b"<!DOCTYPE html", 0, "html", "HTML", [".html", ".htm"]),
    
    # Данные
    (b"PAR1", 0, "data", "Parquet", [".parquet"]),
]

# Папки для различных типов файлов
FOLDER_MAP = {
    "pdf": "pdf",
    "zip": "archives",
    "ole2": "office",
    "rtf": "office",
    "office": "office",
    "images": "images",
    "video": "video",
    "archives": "archives",
    "sqlite": "sqlite",
    "binary": "binary",
    "xml": "data",
    "html": "html",
    "data": "data",
}

# ============================================================
# 2. Функция определения реального формата
# ============================================================

def detect_real_format(file_path: Path) -> Tuple[str, str, Optional[str]]:
    """
    Определяет реальный формат файла по магическим байтам.
    
    Returns:
        (folder_name, format_label, spoof_note)
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(512)
    except (OSError, IOError):
        return "unknown", "ERROR_READ", "не удалось прочитать"
    
    # 1. Проверка magic bytes
    for magic, offset, base_folder, label, expected_exts in MAGIC_SIGNATURES:
        if len(header) > offset + len(magic) and header[offset:offset + len(magic)] == magic:
            folder = FOLDER_MAP.get(base_folder, base_folder)
            
            # Проверка на подмену (расширение не соответствует ожидаемым)
            ext = file_path.suffix.lower()
            spoof_note = None
            if expected_exts and ext not in expected_exts and ext != "":
                spoof_note = f"расширение '{ext}' не соответствует формату {label}"
            
            return folder, label, spoof_note
    
    # 2. Проверка на HTML/текст в первых байтах
    try:
        text_sample = header[:200].decode("utf-8", errors="ignore").lower()
        if "<html" in text_sample or "<!doctype html" in text_sample:
            return "html", "HTML (detected)", None
        # Проверка на JSON
        if text_sample.lstrip().startswith("{") or text_sample.lstrip().startswith("["):
            if '"' in text_sample and ':' in text_sample:
                return "data", "JSON", None
    except Exception:
        pass
    
    # 3. Проверка на текстовый файл (высокий процент печатаемых символов)
    try:
        printable = sum(1 for b in header if 32 <= b <= 126 or b in (9, 10, 13))
        ratio = printable / max(len(header), 1)
        if ratio > 0.7:
            return "text", "TEXT", None
    except Exception:
        pass
    
    # 4. Fallback — по расширению
    ext = file_path.suffix.lower()
    if ext in (".pdf"):
        return "pdf", f"PDF (по расширению, неподтверждённый)", "возможно подмена"
    if ext in (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods"):
        return "office", f"{ext[1:].upper()} (по расширению)", "возможно подмена"
    if ext in (".html", ".htm"):
        return "html", "HTML (по расширению)", "возможно подмена"
    if ext in (".txt", ".md", ".rst", ".log"):
        return "text", "TEXT", None
    if ext in (".csv", ".tsv"):
        return "data", f"{ext[1:].upper()}", None
    if ext in (".json", ".jsonl"):
        return "data", "JSON", None
    if ext in (".xml"):
        return "data", "XML", None
    if ext in (".parquet"):
        return "data", "PARQUET", None
    if ext in (".sqlite", ".db"):
        return "sqlite", "SQLite", None
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"):
        return "images", ext[1:].upper(), None
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        return "video", ext[1:].upper(), None
    if ext in (".zip", ".7z", ".rar", ".gz"):
        return "archives", ext[1:].upper(), None
    
    return "unknown", "UNKNOWN", "неопределённый формат"


# ============================================================
# 3. Обработка ZIP-архивов (опциональное извлечение)
# ============================================================

def extract_archive(archive_path: Path, dest_folder: Path, report: Dict) -> int:
    """Извлекает ZIP-архив, возвращает количество извлечённых файлов."""
    extracted = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                if member.filename.endswith("/"):
                    continue
                # Безопасное имя файла (только basename)
                safe_name = os.path.basename(member.filename)
                if not safe_name:
                    continue
                
                target = dest_folder / safe_name
                # Разрешение коллизий
                counter = 1
                orig_target = target
                while target.exists():
                    stem = orig_target.stem
                    target = orig_target.with_name(f"{stem}_{counter}{orig_target.suffix}")
                    counter += 1
                
                with open(target, "wb") as out:
                    out.write(zf.read(member.filename))
                extracted += 1
                
                # Рекурсивно определяем формат извлечённого файла
                sub_folder, sub_label, sub_spoof = detect_real_format(target)
                report[target.name] = {
                    "original_path": str(archive_path),
                    "extracted_from": archive_path.name,
                    "folder": sub_folder,
                    "format": sub_label,
                    "spoof": sub_spoof,
                }
    except Exception as e:
        log.warning(f"  Не удалось извлечь {archive_path.name}: {e}")
    return extracted


# ============================================================
# 4. Рекурсивный сбор всех файлов
# ============================================================

def get_all_files_recursive(root_dir: Path, exclude_dirs: Set[str] = None) -> List[Path]:
    """
    Рекурсивно собирает все файлы во всех подпапках.
    
    Args:
        root_dir: Корневая директория
        exclude_dirs: Набор имён папок для исключения
    
    Returns:
        Список путей к файлам
    """
    if exclude_dirs is None:
        exclude_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "pdn_reports", "pdn_scan_results"}
    
    all_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
        # Фильтруем исключаемые директории
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith("_extracted_")]
        
        for fn in filenames:
            file_path = Path(dirpath) / fn
            if file_path.is_file():
                all_files.append(file_path)
    
    return all_files


# ============================================================
# 5. Основная функция реорганизации
# ============================================================

def organize_dataset_recursive(
    root_dir: Path,
    output_dir: Path = None,
    mode: str = "move",
    dry_run: bool = False,
    extract_archives: bool = False,
    move_unknown: bool = True,
    preserve_original_names: bool = True,
) -> Dict:
    """
    Рекурсивно реорганизует датасет по папкам реального формата.
    
    Args:
        root_dir: Исходная директория
        output_dir: Целевая директория (для copy/symlink)
        mode: "move", "copy", "symlink"
        dry_run: Только показать, что будет сделано
        extract_archives: Извлекать ZIP-архивы
        move_unknown: Перемещать неизвестные файлы
        preserve_original_names: Сохранять оригинальные имена (иначе использовать хеш)
    
    Returns:
        Словарь со статистикой и отчётом
    """
    if mode == "move":
        target_root = root_dir
        log.info(f"Режим: ПЕРЕМЕЩЕНИЕ (файлы будут переложены внутри {root_dir})")
    else:
        if output_dir is None:
            raise ValueError("Для режимов copy/symlink необходимо указать output_dir")
        target_root = output_dir
        log.info(f"Режим: {mode.upper()} в {target_root}")
    
    if dry_run:
        log.info("РЕЖИМ DRY-RUN: изменения не будут применены")
    
    # Собираем все файлы рекурсивно
    log.info(f"Рекурсивное сканирование {root_dir}...")
    all_files = get_all_files_recursive(root_dir)
    log.info(f"Найдено файлов: {len(all_files)}")
    
    # Статистика
    stats = defaultdict(lambda: {"count": 0, "size": 0, "spoofed": 0})
    report = {
        "scan_date": datetime.now().isoformat(),
        "source": str(root_dir),
        "target": str(target_root),
        "mode": mode,
        "files": {},
        "summary": {},
    }
    
    files_processed = 0
    name_counter: Dict[str, int] = defaultdict(int)  # Для разрешения коллизий
    
    # Обрабатываем каждый файл
    for i, src_path in enumerate(all_files, 1):
        if i % 1000 == 0:
            log.info(f"  Прогресс: {i}/{len(all_files)}")
        
        # Определяем реальный формат
        folder, label, spoof_note = detect_real_format(src_path)
        is_spoofed = spoof_note is not None
        
        # Пропускаем unknown, если не нужно их перемещать
        if folder == "unknown" and not move_unknown:
            log.debug(f"  Пропуск (unknown): {src_path}")
            continue
        
        # Получаем размер файла
        try:
            src_size = src_path.stat().st_size
        except OSError:
            src_size = 0
        
        # Формируем целевое имя
        if preserve_original_names:
            base_name = src_path.name
        else:
            import hashlib
            base_name = hashlib.md5(str(src_path).encode()).hexdigest()[:16] + src_path.suffix
        
        # Разрешение коллизий имён
        dest_name = base_name
        counter = 1
        while True:
            dest_path = target_root / folder / dest_name
            if not dry_run and dest_path.exists():
                stem = Path(base_name).stem
                ext = Path(base_name).suffix
                dest_name = f"{stem}_{counter}{ext}"
                counter += 1
            else:
                break
        
        dest_path = target_root / folder / dest_name
        
        # Действие
        action = None
        if dry_run:
            action = f"[DRY-RUN] {mode} {src_path} -> {dest_path}"
            files_processed += 1
        else:
            # Создаём целевую папку
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if mode == "move":
                    shutil.move(str(src_path), str(dest_path))
                    action = "перемещён"
                elif mode == "copy":
                    shutil.copy2(str(src_path), str(dest_path))
                    action = "скопирован"
                elif mode == "symlink":
                    try:
                        # Относительная симлинка
                        rel_path = os.path.relpath(src_path, dest_path.parent)
                        dest_path.symlink_to(rel_path)
                        action = "симлинк"
                    except Exception:
                        # Fallback: копируем
                        shutil.copy2(str(src_path), str(dest_path))
                        action = "скопирован (fallback)"
                
                files_processed += 1
            except Exception as e:
                log.error(f"  Ошибка при обработке {src_path.name}: {e}")
                continue
            
            # Извлечение архивов (опционально)
            if extract_archives and folder == "archives" and src_path.suffix.lower() == ".zip":
                extracted_dir = target_root / folder / f"_extracted_{dest_path.stem}"
                extracted_dir.mkdir(exist_ok=True)
                extracted_count = extract_archive(dest_path, extracted_dir, report)
                log.info(f"    Извлечено {extracted_count} файлов из {src_path.name}")
        
        # Логируем подмены
        if is_spoofed and not dry_run:
            log.info(f"    ⚠️  ПОДМЕНА: {src_path.name} (реальный: {folder}/{label}, {spoof_note})")
        
        # Статистика
        stats[folder]["count"] += 1
        stats[folder]["size"] += src_size
        if is_spoofed:
            stats[folder]["spoofed"] += 1
        
        # Сохраняем в отчёт
        report["files"][str(dest_path) if not dry_run else str(src_path)] = {
            "original_path": str(src_path),
            "target_path": str(dest_path) if not dry_run else None,
            "size_bytes": src_size,
            "detected_folder": folder,
            "detected_format": label,
            "is_spoofed": is_spoofed,
            "spoof_note": spoof_note,
            "action": action,
        }
    
    # Формируем сводку
    total_size = 0
    for folder in sorted(stats.keys()):
        cnt = stats[folder]["count"]
        sz = stats[folder]["size"]
        spf = stats[folder]["spoofed"]
        total_size += sz
        report["summary"][folder] = {
            "count": cnt,
            "size_bytes": sz,
            "size_mb": round(sz / (1024 * 1024), 2),
            "spoofed_count": spf,
        }
    
    report["summary"]["total"] = {
        "files_processed": files_processed,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "dry_run": dry_run,
    }
    
    # Выводим статистику
    log.info("\n" + "=" * 70)
    log.info("СТАТИСТИКА РЕОРГАНИЗАЦИИ")
    log.info("=" * 70)
    log.info(f"Обработано файлов: {files_processed}")
    
    log.info("\nРаспределение по реальным форматам:")
    for folder in sorted(stats.keys()):
        cnt = stats[folder]["count"]
        sz_mb = stats[folder]["size"] / (1024 * 1024)
        spf = stats[folder]["spoofed"]
        spoof_mark = f" (из них с подменой: {spf})" if spf else ""
        log.info(f"  {folder}/: {cnt} файлов, {sz_mb:.1f} МБ{spoof_mark}")
    
    log.info(f"\nИтого: {files_processed} файлов, {total_size / (1024*1024):.1f} МБ")
    
    if dry_run:
        log.info("\nДля применения изменений запустите без --dry-run")
    
    return report


# ============================================================
# 6. Функция верификации
# ============================================================

def verify_organization(target_dir: Path, report_file: Path = None) -> Dict:
    """Проверяет правильность организации."""
    log.info(f"\nВерификация {target_dir}...")
    
    mismatches = defaultdict(list)
    folder_stats = defaultdict(lambda: {"total": 0, "correct": 0, "size": 0})
    
    for folder in os.listdir(target_dir):
        folder_path = target_dir / folder
        if not folder_path.is_dir() or folder.startswith("_"):
            continue
        
        for file_path in folder_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            real_folder, label, note = detect_real_format(file_path)
            folder_stats[folder]["total"] += 1
            folder_stats[folder]["size"] += file_path.stat().st_size
            
            # Проверка соответствия
            expected_folders = [folder]
            if folder == "office" and real_folder in ("office", "rtf", "ole2"):
                folder_stats[folder]["correct"] += 1
            elif folder == "data" and real_folder in ("data", "xml", "json"):
                folder_stats[folder]["correct"] += 1
            elif folder == real_folder:
                folder_stats[folder]["correct"] += 1
            else:
                mismatches[folder].append((file_path.name, real_folder, label))
    
    log.info("\nРезультаты верификации:")
    for folder in sorted(folder_stats.keys()):
        stats = folder_stats[folder]
        acc = stats["correct"] / max(stats["total"], 1) * 100
        sz_mb = stats["size"] / (1024 * 1024)
        status = "✓" if acc > 95 else "⚠️"
        log.info(f"  {status} {folder}/: {stats['correct']}/{stats['total']} правильных ({acc:.1f}%), {sz_mb:.1f} МБ")
        
        if mismatches[folder]:
            log.info(f"       Несоответствия:")
            for name, real_f, real_l in mismatches[folder][:5]:
                log.info(f"         - {name} → реальный: {real_f} ({real_l})")
            if len(mismatches[folder]) > 5:
                log.info(f"         ... и ещё {len(mismatches[folder]) - 5}")
    
    return mismatches


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Рекурсивная реорганизация датасета по реальному формату файлов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Показать, что будет перемещено (dry-run)
  python organize_by_real_format_recursive.py /path/to/dataset --dry-run

  # Реально переместить файлы в папки по формату (внутри исходной директории)
  python organize_by_real_format_recursive.py /path/to/dataset --mode move

  # Скопировать в новую структуру (сохраняя оригиналы)
  python organize_by_real_format_recursive.py /path/to/dataset --mode copy --output /path/to/organized

  # Извлечь ZIP-архивы в процессе
  python organize_by_real_format_recursive.py /path/to/dataset --mode copy --output /organized --extract-archives

  # Верифицировать уже организованную структуру
  python organize_by_real_format_recursive.py /path/to/organized --verify

  # Сохранить отчёт в JSON
  python organize_by_real_format_recursive.py /path/to/dataset --mode move --report report.json
"""
    )
    parser.add_argument("directory", help="Путь к датасету")
    parser.add_argument("--mode", choices=["move", "copy", "symlink"], default="move",
                        help="Режим: move (переместить), copy (скопировать), symlink (симлинки)")
    parser.add_argument("--output", "-o", help="Целевая директория (для copy/symlink)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать, что будет сделано (без изменений)")
    parser.add_argument("--extract-archives", action="store_true",
                        help="Извлекать ZIP-архивы в подпапки _extracted_/")
    parser.add_argument("--no-unknown", action="store_true",
                        help="Не перемещать файлы с неопределённым форматом")
    parser.add_argument("--preserve-names", action="store_true", default=True,
                        help="Сохранять оригинальные имена файлов (по умолчанию)")
    parser.add_argument("--verify", action="store_true",
                        help="Верифицировать уже организованную структуру")
    parser.add_argument("--report", "-r", type=str,
                        help="Сохранить отчёт в JSON файл")
    
    args = parser.parse_args()
    
    root = Path(args.directory).resolve()
    if not root.exists():
        log.error(f"Директория не найдена: {root}")
        return 1
    
    if args.verify:
        verify_organization(root)
        return 0
    
    output_dir = None
    if args.mode in ("copy", "symlink"):
        if not args.output:
            log.error("Для режимов copy/symlink необходимо указать --output")
            return 1
        output_dir = Path(args.output).resolve()
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    report = organize_dataset_recursive(
        root_dir=root,
        output_dir=output_dir,
        mode=args.mode,
        dry_run=args.dry_run,
        extract_archives=args.extract_archives,
        move_unknown=not args.no_unknown,
        preserve_original_names=args.preserve_names,
    )
    
    # Сохраняем отчёт
    if args.report and not args.dry_run:
        report_path = Path(args.report)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"Отчёт сохранён: {report_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())