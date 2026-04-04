"""
modules/iris/ingestion.py

Iris ingestion and deduplication using IrisDB (sqlite3, no ORM).
"""
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import time

from .db import IrisDB
from .config import IrisConfig

logger = logging.getLogger(__name__)

# --- Helpers ---
def get_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {'.jpg','.jpeg','.png','.gif','.bmp','.tiff','.tif',
               '.webp','.heic','.heif','.raw','.nef','.cr2','.arw',
               '.dng','.orf','.sr2'}: return 'image'
    if ext in {'.mp4','.mov','.avi','.mkv','.webm','.flv','.wmv',
               '.m4v','.mpg','.mpeg','.3gp','.mts','.m2ts'}: return 'video'
    if ext in {'.mp3','.wav','.flac','.m4a','.aac','.ogg','.wma'}: return 'audio'
    return 'other'

# --- Dataclasses ---
@dataclass
class FileInfo:
    path: Path
    size: int
    mtime: datetime
    hash: str
    perceptual_hash: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    error: Optional[str] = None

# --- Duplicate Detection ---
class DuplicateDetector:
    def __init__(self, db: IrisDB):
        self.db = db

    async def compute_file_hash(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            raise

    async def compute_perceptual_hash(self, file_path: Path) -> Optional[str]:
        try:
            import imagehash
            from PIL import Image
            with Image.open(file_path) as img:
                if img.mode not in ["RGB", "L"]:
                    img = img.convert("RGB")
                ahash = str(imagehash.average_hash(img))
                phash = str(imagehash.phash(img))
                return f"{ahash}:{phash}"
        except ImportError:
            logger.warning("imagehash not installed, perceptual hashing disabled")
            return None
        except Exception as e:
            logger.error(f"Error computing perceptual hash for {file_path}: {e}")
            return None

    async def find_duplicates(self, file_path: Path, file_hash: str, perceptual_hash: Optional[str] = None) -> List[Tuple[int, str, float]]:
        # 0. If exact path exists, treat as already ingested
        if self.db.file_exists(str(file_path)):
            return [(0, str(file_path), 1.0)]
        duplicates = []
        # 1. Exact hash
        if self.db.file_exists_by_hash(file_hash):
            duplicates.append((0, str(file_path), 1.0))
        # 2. Perceptual hash (not implemented in DB, stub)
        # Could be extended to check visually similar files
        return duplicates

# --- File Ingestor ---
class FileIngestor:
    def __init__(self, config: IrisConfig, db: IrisDB):
        self.config = config
        self.db = db
        self.duplicate_detector = DuplicateDetector(db)
        self.stats: Dict[str, int] = {"ingested": 0, "duplicates_skipped": 0, "errors": 0, "total_size": 0}
        self.retry_queue: List[Tuple[Path, int]] = []

    async def process_directory(self, directory: Path, recursive: bool = True) -> Dict:
        logger.info(f"Processing directory: {directory}")
        self.stats = {"ingested": 0, "duplicates_skipped": 0, "errors": 0, "total_size": 0}
        self.retry_queue = []
        file_paths = self._find_media_files(directory, recursive)
        logger.info(f"Found {len(file_paths)} files to process")
        batch_size = self.config.batch_size
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(file_paths) + batch_size - 1) // batch_size}")
            await self._process_batch(batch)
        while self.retry_queue:
            await self.retry_failed_files()
        logger.info(
            "[INGEST] SUMMARY — ingested=%d, duplicates=%d, errors=%d, total_size=%d bytes",
            self.stats["ingested"], self.stats["duplicates_skipped"], self.stats["errors"], self.stats["total_size"]
        )
        return self.stats.copy()

    def _find_media_files(self, directory: Path, recursive: bool) -> List[Path]:
        media_extensions = {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
            ".webp", ".heic", ".heif", ".raw", ".nef", ".cr2", ".arw",
            ".dng", ".orf", ".sr2",
            ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
            ".m4v", ".mpg", ".mpeg", ".3gp", ".mts", ".m2ts",
            ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma",
        }
        files: List[Path] = []
        if recursive:
            for ext in media_extensions:
                files.extend(directory.rglob(f"*{ext}"))
                files.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in media_extensions:
                files.extend(directory.glob(f"*{ext}"))
                files.extend(directory.glob(f"*{ext.upper()}"))
        filtered_files = [f for f in files if f.is_file() and os.access(f, os.R_OK) and not f.name.startswith(('.', '~')) and f.name not in ["Thumbs.db", "desktop.ini", ".DS_Store"]]
        seen = set()
        unique_files = []
        for fp in filtered_files:
            key = str(fp.resolve())
            if key not in seen:
                seen.add(key)
                unique_files.append(fp)
        unique_files.sort(key=lambda x: x.stat().st_mtime)
        return unique_files

    async def _process_batch(self, file_paths: List[Path]):
        tasks = [self._process_single_file(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {file_path}: {result}")
                self.retry_queue.append((file_path, 1))
                self.stats["errors"] += 1
                continue
            if result is not None:
                self._update_stats(result)

    async def retry_failed_files(self):
        if not self.retry_queue:
            return
        current_queue = self.retry_queue
        self.retry_queue = []
        for file_path, attempt in current_queue:
            delay_map = {1: 3, 2: 10, 3: 30}
            delay = delay_map.get(attempt, 30)
            logger.info(f"[RETRY] Attempt {attempt} for {file_path} after {delay}s")
            await asyncio.sleep(delay)
            try:
                result = await self._process_single_file(file_path)
                if result is not None:
                    self._update_stats(result)
            except Exception as e:
                logger.error(f"[RETRY] Error processing {file_path}: {e}")
                if attempt < 3:
                    self.retry_queue.append((file_path, attempt + 1))
                else:
                    logger.error(f"[RETRY] Permanent failure for {file_path}")
                    self.stats["errors"] += 1

    async def _process_single_file(self, file_path: Path) -> Optional[FileInfo]:
        start = time.time()
        logger.info(f"[INGEST] START {file_path}")
        if self.db.file_exists(str(file_path)):
            logger.info(f"[INGEST] SKIP (already in DB) {file_path}")
            stat = file_path.stat()
            duration = (time.time() - start) * 1000
            file_info = FileInfo(
                path=file_path,
                size=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime),
                hash="",
                is_duplicate=True,
                duplicate_of=str(file_path),
            )
            logger.info(f"[INGEST] DONE {file_path} ({duration:.1f} ms) [DUPLICATE-PATH]")
            return file_info
        try:
            stat = file_path.stat()
            file_info = FileInfo(
                path=file_path,
                size=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime),
                hash="",
            )
            file_info.hash = await self.duplicate_detector.compute_file_hash(file_path)
            file_info.perceptual_hash = await self.duplicate_detector.compute_perceptual_hash(file_path)
            logger.info(f"[INGEST] HASH  {file_info.hash[:12]}")
            logger.info(f"[INGEST] PHASH {file_info.perceptual_hash}")
            duplicates = await self.duplicate_detector.find_duplicates(file_path, file_info.hash, file_info.perceptual_hash)
            if duplicates:
                file_info.is_duplicate = True
                file_info.duplicate_of = duplicates[0][1]
                logger.info(f"[INGEST] DUPLICATE {file_path} → {file_info.duplicate_of}")
                duration = (time.time() - start) * 1000
                logger.info(f"[INGEST] DONE {file_path} ({duration:.1f} ms) [DUPLICATE]")
                return file_info
            logger.info(f"[INGEST] UNIQUE {file_path}")
            await self._ingest_file(file_info)
            duration = (time.time() - start) * 1000
            logger.info(f"[INGEST] DONE {file_path} ({duration:.1f} ms) [INGESTED]")
            return file_info
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return FileInfo(
                path=file_path,
                size=0,
                mtime=datetime.now(),
                hash="",
                error=str(e),
            )

    async def _ingest_file(self, file_info: FileInfo):
        file_type = get_file_type(file_info.path)
        mime_type = self._guess_mime_type(file_info.path)
        file_id = self.db.insert_file(
            str(file_info.path),
            file_info.hash,
            file_info.perceptual_hash,
            file_info.size,
            file_type,
            mime_type,
        )
        self.db.enqueue(file_id, task_type="analyze", priority=0)
        logger.info(f"Ingested file: {file_info.path} (ID: {file_id})")

    def _guess_mime_type(self, file_path: Path) -> str:
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
            ".heic": "image/heic",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
        }
        ext = file_path.suffix.lower()
        return mime_types.get(ext, "application/octet-stream")

    def _update_stats(self, file_info: FileInfo):
        if file_info.error:
            self.stats["errors"] += 1
        elif file_info.is_duplicate:
            self.stats["duplicates_skipped"] += 1
        else:
            self.stats["ingested"] += 1
            self.stats["total_size"] += file_info.size
