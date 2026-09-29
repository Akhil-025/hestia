# tests/test_iris_exif_and_quota.py
"""
Tests for EXIF extraction and search (backlog #74) and the storage-budget
guard (backlog #80).
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ExifTags

from modules.iris.analyser import _extract_exif, _gps_to_decimal
from modules.iris.db import IrisDB
from modules.iris.config import IrisConfig
from modules.iris.iris_engine import IrisEngine


def make_exif_image(date=None, make=None, model=None, gps=None):
    img = Image.new("RGB", (10, 10), color="blue")
    exif = Image.Exif()
    if date:
        exif[306] = date  # DateTime
    if make:
        exif[271] = make  # Make
    if model:
        exif[272] = model  # Model
    if gps:
        exif[ExifTags.IFD.GPSInfo] = gps
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# _extract_exif
# ---------------------------------------------------------------------------

def test_extracts_date_taken():
    img = make_exif_image(date="2024:03:15 10:30:00")
    result = _extract_exif(img)
    assert result["date_taken"] == "2024-03-15T10:30:00"


def test_extracts_camera_make_and_model():
    img = make_exif_image(make="Canon", model="EOS R5")
    result = _extract_exif(img)
    assert result["camera_make"] == "Canon"
    assert result["camera_model"] == "EOS R5"


def test_extracts_gps_coordinates_with_correct_hemisphere_signs():
    gps = {1: "N", 2: (37.0, 46.0, 30.0), 3: "W", 4: (122.0, 25.0, 10.0)}
    img = make_exif_image(gps=gps)
    result = _extract_exif(img)
    assert result["gps_lat"] > 0   # N is positive
    assert result["gps_lon"] < 0   # W is negative


def test_southern_and_eastern_hemispheres_flip_correctly():
    gps = {1: "S", 2: (33.0, 52.0, 0.0), 3: "E", 4: (151.0, 12.0, 0.0)}
    img = make_exif_image(gps=gps)
    result = _extract_exif(img)
    assert result["gps_lat"] < 0   # S is negative
    assert result["gps_lon"] > 0   # E is positive


def test_image_with_no_exif_returns_empty_dict():
    img = Image.new("RGB", (10, 10))
    assert _extract_exif(img) == {}


def test_malformed_date_is_skipped_not_guessed():
    img = make_exif_image(date="not a real date")
    result = _extract_exif(img)
    assert "date_taken" not in result


def test_extraction_never_raises_on_a_broken_image_object():
    class _Explodes:
        def getexif(self):
            raise RuntimeError("corrupt EXIF block")
    assert _extract_exif(_Explodes()) == {}


def test_gps_to_decimal_handles_missing_input():
    assert _gps_to_decimal(None, "N") is None
    assert _gps_to_decimal((1, 2, 3), None) is None


def test_gps_to_decimal_handles_malformed_coordinates():
    assert _gps_to_decimal(("not", "a", "number"), "N") is None


# ---------------------------------------------------------------------------
# IrisDB EXIF storage and search
# ---------------------------------------------------------------------------

def make_db(tmp_path):
    return IrisDB(str(tmp_path / "iris.db"))


def insert_file(db, path="a.jpg", file_hash="h1"):
    return db.insert_file(path, file_hash, perceptual_hash="p" + file_hash, file_size=100, file_type="image", mime_type="image/jpeg")


def test_update_file_exif_round_trips(tmp_path):
    db = make_db(tmp_path)
    file_id = insert_file(db)
    db.update_file_exif(
        file_id, date_taken="2024-03-15T10:30:00", gps_lat=37.5, gps_lon=-122.4,
        camera_make="Canon", camera_model="EOS R5",
    )
    row = db.get_file(file_id)
    assert row["camera_make"] == "Canon"
    assert row["gps_lat"] == 37.5


def test_search_by_exif_date_range(tmp_path):
    db = make_db(tmp_path)
    id_a = insert_file(db, "a.jpg", "h1")
    id_b = insert_file(db, "b.jpg", "h2")
    db.update_file_exif(id_a, date_taken="2024-01-01T00:00:00")
    db.update_file_exif(id_b, date_taken="2024-06-01T00:00:00")

    results = db.search_files_by_exif(date_from="2024-03-01", date_to="2024-12-31")
    ids = {r["id"] for r in results}
    assert ids == {id_b}


def test_search_by_exif_camera_matches_make_or_model(tmp_path):
    db = make_db(tmp_path)
    id_a = insert_file(db, "a.jpg", "h1")
    db.update_file_exif(id_a, camera_make="Canon", camera_model="EOS R5")

    assert len(db.search_files_by_exif(camera="Canon")) == 1
    assert len(db.search_files_by_exif(camera="R5")) == 1
    assert len(db.search_files_by_exif(camera="Nikon")) == 0


def test_search_by_exif_has_location_filter(tmp_path):
    db = make_db(tmp_path)
    id_with_gps = insert_file(db, "a.jpg", "h1")
    id_without_gps = insert_file(db, "b.jpg", "h2")
    db.update_file_exif(id_with_gps, gps_lat=1.0, gps_lon=2.0)

    with_location = db.search_files_by_exif(has_location=True)
    assert {r["id"] for r in with_location} == {id_with_gps}

    without_location = db.search_files_by_exif(has_location=False)
    assert {r["id"] for r in without_location} == {id_without_gps}


def test_search_by_exif_with_no_filters_returns_everything(tmp_path):
    db = make_db(tmp_path)
    insert_file(db, "a.jpg", "h1")
    insert_file(db, "b.jpg", "h2")
    assert len(db.search_files_by_exif()) == 2


def test_search_by_exif_combines_filters_with_and(tmp_path):
    db = make_db(tmp_path)
    id_a = insert_file(db, "a.jpg", "h1")
    id_b = insert_file(db, "b.jpg", "h2")
    db.update_file_exif(id_a, date_taken="2024-01-01T00:00:00", camera_make="Canon")
    db.update_file_exif(id_b, date_taken="2024-01-01T00:00:00", camera_make="Nikon")

    results = db.search_files_by_exif(date_from="2024-01-01", camera="Canon")
    assert {r["id"] for r in results} == {id_a}


# ---------------------------------------------------------------------------
# Storage quota guard (#80)
# ---------------------------------------------------------------------------

def make_iris_config(tmp_path, quota_bytes=None):
    return IrisConfig(
        db_path=str(tmp_path / "iris.db"),
        source_dir=str(tmp_path / "source"),
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        chroma_dir=str(tmp_path / "chroma"),
        storage_quota_bytes=quota_bytes,
    )


def make_bare_engine(tmp_path, quota_bytes=None):
    """An IrisEngine with its config/db swapped for isolated test doubles,
    bypassing the real persisted data/iris/iris.db the module-level
    default config points at."""
    engine = IrisEngine.__new__(IrisEngine)
    engine.config = make_iris_config(tmp_path, quota_bytes)
    engine.db = IrisDB(engine.config.db_path)
    return engine


def test_no_quota_configured_never_warns(tmp_path):
    engine = make_bare_engine(tmp_path, quota_bytes=None)
    os.makedirs(engine.config.source_dir, exist_ok=True)
    with open(os.path.join(engine.config.source_dir, "big.jpg"), "wb") as f:
        f.write(b"x" * 1000)
    assert engine.check_storage_quota() is None


def test_quota_exceeded_returns_a_warning(tmp_path):
    engine = make_bare_engine(tmp_path, quota_bytes=500)
    os.makedirs(engine.config.source_dir, exist_ok=True)
    with open(os.path.join(engine.config.source_dir, "big.jpg"), "wb") as f:
        f.write(b"x" * 1000)  # bigger than the 500-byte quota
    warning = engine.check_storage_quota()
    assert warning is not None
    assert warning["exceeds_quota"] is True
    assert warning["projected_bytes"] >= 1000


def test_quota_not_exceeded_returns_none(tmp_path):
    engine = make_bare_engine(tmp_path, quota_bytes=10_000)
    os.makedirs(engine.config.source_dir, exist_ok=True)
    with open(os.path.join(engine.config.source_dir, "small.jpg"), "wb") as f:
        f.write(b"x" * 100)
    assert engine.check_storage_quota() is None


def test_quota_accounts_for_already_ingested_bytes(tmp_path):
    engine = make_bare_engine(tmp_path, quota_bytes=1000)
    engine.db.insert_file("existing.jpg", "hash1", perceptual_hash="phash1", file_size=900, file_type="image", mime_type="image/jpeg")
    os.makedirs(engine.config.source_dir, exist_ok=True)
    with open(os.path.join(engine.config.source_dir, "new.jpg"), "wb") as f:
        f.write(b"x" * 200)  # 900 + 200 > 1000
    warning = engine.check_storage_quota()
    assert warning is not None
    assert warning["current_bytes"] == 900


def test_quota_check_on_a_missing_directory_does_not_raise(tmp_path):
    engine = make_bare_engine(tmp_path, quota_bytes=100)
    assert engine.check_storage_quota(str(tmp_path / "does_not_exist")) is None
