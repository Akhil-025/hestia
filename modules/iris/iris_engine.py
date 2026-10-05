"""
modules/iris/iris_engine.py

IrisEngine: single entry point for Iris image/video/audio ingestion and search.
"""
from modules.base import BaseModule  
import logging
import asyncio
import json
from .config import get_config, IrisConfig
from pathlib import Path
from .db import IrisDB
from .ingestion import FileIngestor
from .embeddings import ClipEmbedder, ImageVectorIndex, filter_hits
from . import video as video_mod
from . import detection as detection_mod
from .faces import FaceService, OpenCVFaceBackend
try:
    from .analyser import IrisAnalyser, _load_image_base64
except ImportError:
    IrisAnalyser = None
    _load_image_base64 = None
    logging.warning("[Iris] requests library not available, IrisAnalyser disabled.")

logger = logging.getLogger(__name__)

class IrisEngine(BaseModule):               
    name = "iris"

    def __init__(self, hestia_llm=None, embedder=None, vector_index=None,
                 face_service=None, detector=None, camera_factory=None):
        try:
            self.config: IrisConfig = get_config()
            self.db = IrisDB(self.config.db_path)
            self.ingestor = FileIngestor(self.config, self.db)
            self.hestia_llm = hestia_llm

            # CLIP semantic search (README roadmap item — see embeddings.py).
            # Both are injectable for tests; default to real implementations
            # that degrade to unavailable on their own if torch/chromadb
            # aren't installed, so Iris as a whole never fails to construct
            # because of this.
            self.embedder = embedder if embedder is not None else ClipEmbedder()
            self.vector_index = (
                vector_index if vector_index is not None
                else ImageVectorIndex(self.config.chroma_dir)
            )

            ollama_host = "127.0.0.1"
            ollama_port = 11434
            self.analyser = (
                IrisAnalyser(
                    self.db, ollama_host, ollama_port, "llava:7b",
                    embedder=self.embedder, vector_index=self.vector_index,
                    video_frames=getattr(self.config, "video_frames", 4),
                    work_dir=self.config.cache_dir,
                )
                if IrisAnalyser else None
            )

            # Face grouping (#72) and camera object detection (#77). Both are
            # off by default (biometric data / opens the webcam), both load
            # their models lazily, and both are injectable for tests, so
            # neither can stop Iris constructing.
            self.faces = face_service if face_service is not None else FaceService(
                self.db,
                OpenCVFaceBackend(
                    getattr(self.config, "face_detector_model", ""),
                    getattr(self.config, "face_recognizer_model", ""),
                    min_face_size=getattr(self.config, "face_min_size", 40),
                ),
                enabled=getattr(self.config, "faces_enabled", False),
                threshold=getattr(self.config, "face_match_threshold", 0.45),
                min_cluster_size=getattr(self.config, "face_min_cluster_size", 2),
            )
            self.detector = detector if detector is not None else detection_mod.YoloDetector(
                getattr(self.config, "detector_model", "yolov8n.pt"),
                getattr(self.config, "detector_confidence", 0.4),
            )
            self._camera_factory = camera_factory if camera_factory is not None else (
                lambda: detection_mod.Camera(getattr(self.config, "camera_index", 0))
            )
            logger.info("[Iris] Engine ready")
        except Exception as e:
            logger.error(f"[Iris] Engine init failed: {e}")
            raise


    def can_handle(self, intent: str) -> bool:   
        # Accepts both the full prefixed intent name (as emitted by the NLU
        # / used by Hecate's Tier-1.5 direct routing) and the stripped form
        # produced by HestiaOrchestrator._strip_module_prefix() before
        # dispatch — without this, "iris_search" gets stripped to "search"
        # and can_handle() would reject it, silently falling back to chat.
        return intent in {
            "iris_search", "iris_ingest", "iris_analyse", "iris_status", "iris_query",
            "search", "ingest", "analyse", "status", "query",
            # backlog #73, #78, #79, #76 — cleanup/organisation/comparison
            # tools beyond plain ingest/search.
            "iris_find_duplicates", "find_duplicates",
            "iris_correct_caption", "correct_caption",
            "iris_organize_albums", "organize_albums",
            "iris_compare_photos", "compare_photos",
            # backlog #71 (re-index, find-similar), #72 (faces), #77 (objects)
            "iris_reindex", "reindex",
            "iris_find_similar", "find_similar",
            "iris_scan_faces", "scan_faces",
            "iris_list_people", "list_people",
            "iris_name_person", "name_person",
            "iris_find_person", "find_person",
            "iris_forget_faces", "forget_faces",
            "iris_detect_objects", "detect_objects",
        }

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        raw = entities.get("raw_query", context.get("raw_query", "")).lower()

        # Normalise: accept both the "iris_"-prefixed intent (as emitted by
        # the NLU / used by Hecate's Tier-1.5 direct routing) and the
        # stripped form HestiaOrchestrator._strip_module_prefix() actually
        # passes to handle() for every non-trigger-tier dispatch ("iris_"
        # is in the orchestrator's known prefix list). Branching on the
        # prefixed form only, as this used to, meant the normal dispatch
        # path — intent="search"/"ingest"/"analyse"/"status" — never
        # matched anything here and silently fell through to the blank,
        # 0.0-confidence response below, even though can_handle() had
        # already reported True for that exact intent.
        canonical = intent[len("iris_"):] if intent.startswith("iris_") else intent

        if canonical == "ingest":
            stats = self.ingest(force=bool(entities.get("force")))
            if stats.get("exceeds_quota"):
                quota_gb = stats["quota_bytes"] / 1e9
                projected_gb = stats["projected_bytes"] / 1e9
                return {
                    "response": (
                        f"Ingesting this would use about {projected_gb:.1f} GB, "
                        f"over your {quota_gb:.1f} GB storage quota — skipped. "
                        f"Free up space, raise the quota, or ingest anyway."
                    ),
                    "data": stats,
                    "confidence": 0.9,
                }
            return {
                "response": (
                    f"Media ingestion complete. "
                    f"{stats.get('ingested', 0)} files processed, "
                    f"{stats.get('duplicates_skipped', 0)} duplicates skipped."
                ),
                "data": stats,
                "confidence": 1.0,
            }

        elif canonical == "analyse":
            response = self.analyse(limit=20)
            return {"response": response, "data": {}, "confidence": 0.9}

        elif canonical == "status":
            # Previously had no branch at all — can_handle() accepted
            # "status"/"iris_status" but handle() silently dropped it into
            # the blank fallback below.
            return {"response": self.status(), "data": {}, "confidence": 0.9}

        elif canonical in {"search", "query"}:
            # "photos of Mom" — if the query names someone Iris has been told
            # about (#72), answer from the face groups instead.
            person_reply = self._person_search(raw or entities.get("query", ""))
            if person_reply is not None:
                return person_reply
            result = self.search(raw or entities.get("query", ""))
            return {
                "response": result or "No matching media found.",
                "data": {},
                "confidence": 0.85 if result else 0.3,
            }

        elif canonical == "find_duplicates":
            groups = self.find_duplicates()
            if not groups:
                response = "No duplicate or near-duplicate photos found."
            else:
                total_files = sum(len(g["files"]) for g in groups)
                response = (
                    f"Found {len(groups)} duplicate group(s) "
                    f"({total_files} files total). Largest group: "
                    f"{len(groups[0]['files'])} file(s)."
                )
            return {"response": response, "data": {"groups": groups}, "confidence": 0.9}

        elif canonical == "correct_caption":
            file_id = entities.get("file_id")
            caption = entities.get("caption")
            tags = entities.get("tags")
            if file_id is None or (caption is None and tags is None):
                return {
                    "response": "Tell me which photo and what the caption or tags should be.",
                    "data": {}, "confidence": 0.0,
                }
            ok = self.db.correct_caption(int(file_id), caption=caption, tags=tags)
            response = "Updated." if ok else "I couldn't find that photo."
            return {"response": response, "data": {"updated": ok}, "confidence": 0.9 if ok else 0.3}

        elif canonical == "organize_albums":
            albums = self.organize_into_albums()
            if not albums:
                response = "Not enough indexed photos with embeddings to organize into albums yet."
            else:
                response = f"Organized {sum(len(a['files']) for a in albums)} photos into {len(albums)} album(s)."
            return {"response": response, "data": {"albums": albums}, "confidence": 0.85 if albums else 0.4}

        elif canonical == "compare_photos":
            file_id_a = entities.get("file_id_a") or entities.get("file_id_1")
            file_id_b = entities.get("file_id_b") or entities.get("file_id_2")
            if file_id_a is None or file_id_b is None:
                return {
                    "response": "Tell me which two photos to compare.",
                    "data": {}, "confidence": 0.0,
                }
            result = self.describe_change(int(file_id_a), int(file_id_b))
            return {
                "response": result.get("description", "I couldn't compare those photos."),
                "data": result, "confidence": 0.85 if result.get("description") else 0.2,
            }

        elif canonical == "reindex":
            stats = self.reindex_embeddings(_as_int(entities.get("limit"), 200))
            if stats["unavailable"]:
                response = ("Semantic search isn't available right now (the CLIP model or the "
                            "vector index didn't load), so there's nothing to index.")
            else:
                response = (f"Indexed {stats['indexed']} item(s) for semantic search"
                            + (f"; {stats['failed']} couldn't be read" if stats["failed"] else "")
                            + (f"; about {stats['remaining']} still to do, ask again to continue"
                               if stats["remaining"] else "")
                            + ("." if stats["indexed"] or stats["failed"] or stats["remaining"]
                               else "; everything is already indexed."))
            return {"response": response, "data": stats,
                    "confidence": 0.3 if stats["unavailable"] else 0.9}

        elif canonical == "find_similar":
            fid = _as_int(entities.get("file_id"), None)
            if fid is None:
                return {"response": "Tell me which photo to find look-alikes of.",
                        "data": {}, "confidence": 0.0}
            result = self.find_similar(fid, _as_int(entities.get("limit"), 10))
            if result.get("error"):
                return {"response": result["error"], "data": result, "confidence": 0.3}
            if not result["matches"]:
                return {"response": "I didn't find anything similar.", "data": result, "confidence": 0.5}
            lines = [f"Found {len(result['matches'])} similar photo(s):"]
            lines += [f"{i}. {self._describe(r)}" for i, r in enumerate(result["matches"], 1)]
            return {"response": "\n".join(lines), "data": result, "confidence": 0.85}

        elif canonical == "scan_faces":
            stats = self.faces.scan(_as_int(entities.get("limit"), 100))
            if not stats.get("available"):
                return {"response": stats["message"], "data": stats, "confidence": 0.5}
            response = (f"Scanned {stats['scanned']} photo(s) and found {stats['faces_found']} face(s). "
                        f"{stats['new_people']} new group(s), {stats['added_to_existing']} face(s) added "
                        f"to people I already know")
            response += f"; {stats['failed']} photo(s) couldn't be read" if stats["failed"] else ""
            response += f". {stats['remaining']} photo(s) still to scan." if stats["remaining"] else "."
            if stats["new_people"]:
                response += " Say who someone is to name them, e.g. 'Person 3 is Mom'."
            return {"response": response, "data": stats, "confidence": 0.9}

        elif canonical == "list_people":
            blocked = self.faces.unavailable_reason()
            if blocked:
                return {"response": blocked, "data": {}, "confidence": 0.5}
            people = self.faces.list_people()
            if not people:
                return {"response": "I haven't grouped any faces yet. Ask me to scan your photos for faces first.",
                        "data": {"people": []}, "confidence": 0.5}
            lines = [f"I know {len(people)} group(s) of faces:"]
            lines += [f"- {p['label']}: {p['photo_count']} photo(s)" for p in people[:20]]
            if len(people) > 20:
                lines.append(f"…and {len(people) - 20} more.")
            return {"response": "\n".join(lines), "data": {"people": people}, "confidence": 0.9}

        elif canonical == "name_person":
            blocked = self.faces.unavailable_reason()
            if blocked:
                return {"response": blocked, "data": {}, "confidence": 0.5}
            ref, name = entities.get("person"), entities.get("name")
            if ref is None or name is None:
                return {"response": "Tell me which group and who it is, e.g. 'Person 3 is Mom'.",
                        "data": {}, "confidence": 0.0}
            outcome = self.faces.name_person(ref, name)
            if not outcome["ok"]:
                return {"response": outcome["reason"], "data": outcome, "confidence": 0.3}
            response = (f"Merged them: they're all {outcome['label']} now." if outcome["merged"]
                        else f"Got it, that's {outcome['label']}.")
            return {"response": response, "data": outcome, "confidence": 0.9}

        elif canonical == "find_person":
            blocked = self.faces.unavailable_reason()
            if blocked:
                return {"response": blocked, "data": {}, "confidence": 0.5}
            reply = self._person_files_reply(entities.get("person") or entities.get("name"), "")
            return reply or {"response": "I don't know anyone by that name yet.", "data": {}, "confidence": 0.3}

        elif canonical == "forget_faces":
            counts = self.faces.forget_all()
            return {"response": (f"Done. I deleted {counts['faces']} face record(s) and "
                                 f"{counts['people']} person group(s). Your photos are untouched."),
                    "data": counts, "confidence": 0.95}

        elif canonical == "detect_objects":
            result = self.detect_objects(
                file_id=_as_int(entities.get("file_id"), None),
                seconds=_as_float(entities.get("seconds"), None),
            )
            return {"response": result["response"], "data": result.get("data", {}),
                    "confidence": result.get("confidence", 0.8)}

        return {"response": "I'm not sure how to handle that media request.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        try:
            s = self.stats()
            recent = self.db.get_all_files(limit=3)
            recent_captions = [
                f.get("caption", "") for f in recent if f.get("caption")
            ]
            return {
                "iris_total":         s.get("total_files", 0),
                "iris_processed":     s.get("processed", 0),
                "iris_pending":       s.get("pending", 0),
                "iris_recent_captions": recent_captions,
            }
        except Exception:
            return {}

    def check_storage_quota(self, source_dir: str = None) -> "dict | None":
        """
        backlog #80. Returns None if no quota is configured, or if
        ingesting *source_dir* wouldn't push total ingested media past it.
        Otherwise returns a warning dict describing by how much.

        The incoming estimate sums every file under source_dir regardless
        of whether it's already ingested (duplicates get skipped at
        ingest time and don't consume additional space) — this errs
        toward warning a LITTLE early rather than under-estimating and
        letting a quota get blown past silently, which is the one failure
        mode a "guard" must never have.
        """
        quota = getattr(self.config, "storage_quota_bytes", None)
        if not quota:
            return None

        directory = Path(source_dir or self.config.source_dir)
        if not directory.is_dir():
            return None

        try:
            incoming = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file()
            )
        except OSError:
            return None  # can't estimate — don't block ingestion over it

        current = self.db.get_total_ingested_bytes()
        projected = current + incoming
        if projected <= quota:
            return None

        return {
            "exceeds_quota": True,
            "quota_bytes": quota,
            "current_bytes": current,
            "estimated_incoming_bytes": incoming,
            "projected_bytes": projected,
        }

    def ingest(self, source_dir: str = None, force: bool = False) -> dict:
        try:
            dir_to_use = source_dir or self.config.source_dir

            if not force:
                quota_warning = self.check_storage_quota(dir_to_use)
                if quota_warning is not None:
                    return {**quota_warning, "ingested": 0, "duplicates_skipped": 0,
                            "errors": 0, "total_size": 0}

            coro = self.ingestor.process_directory(
                Path(dir_to_use), recursive=True
            )

            try:
                loop = asyncio.get_running_loop()

                # ⚠️ If we're already inside the event loop thread
                if loop.is_running():
                    # Offload to thread to avoid blocking loop
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(coro))
                        return future.result()

            except RuntimeError:
                # No running loop
                return asyncio.run(coro)

        except Exception as e:
            logger.error(f"[Iris] Ingest error: {e}")
            return {
                "ingested": 0,
                "duplicates_skipped": 0,
                "errors": 1,
                "total_size": 0,
            }

    def _semantic_matches(self, query: str, limit: int) -> list[dict]:
        """
        CLIP-embedding-based matches, ranked nearest-first. Returns [] (not
        an error) whenever the embedder/vector_index aren't available or the
        query can't be embedded — this is a ranking *enhancement* over
        caption/tag search, never a hard dependency for search() to work.
        """
        try:
            query_vector = self.embedder.embed_text(query)
            if query_vector is None:
                return []
            hits = self.vector_index.query(query_vector, top_k=limit)
            # #71: optional relevance cut-offs (off unless configured)
            hits = filter_hits(
                hits,
                _as_float(getattr(self.config, "semantic_max_distance", None), None),
                _as_float(getattr(self.config, "semantic_relative_margin", None), None),
            )
            matches = []
            for file_id, _distance in hits:
                record = self.db.get_file(file_id)
                if record:
                    matches.append(record)
            return matches
        except Exception as e:
            logger.warning(f"[Iris] Semantic search failed, falling back to caption/tag: {e}")
            return []

    def search(self, query: str, limit: int = 10) -> "str | None":
        try:
            # #74: date taken / camera / "with location" filters, alongside the
            # caption, tag and semantic search. With no such words in the query
            # this is a no-op and behaves exactly as before.
            from modules.iris.query_filters import parse_photo_query
            pq = parse_photo_query(query)
            exif_rows: list = []
            if pq.active:
                exif_rows = self.db.search_files_by_exif(
                    date_from=pq.date_from, date_to=pq.date_to, camera=pq.camera,
                    has_location=pq.has_location, limit=500,
                )
                if not pq.text:
                    results_semantic, results_caption, results_tags = exif_rows[:limit], [], []
                else:
                    query = pq.text
                    results_semantic = self._semantic_matches(query, limit * 5)
                    results_caption = self.db.search_files_by_caption(query, limit * 5)
                    results_tags = self.db.search_files_by_tags(query, limit * 5)
            else:
                results_semantic = self._semantic_matches(query, limit)
                results_caption = self.db.search_files_by_caption(query, limit)
                results_tags = self.db.search_files_by_tags(query, limit)
            # #77: photos whose detected objects include the query ("laptop").
            # Not for a pure date/camera query, where the words are filters.
            if not (pq.active and not pq.text):
                results_tags = list(results_tags) + self._object_matches(query, limit)
            # Deduplicate by file_path. Semantic hits are listed first so
            # `dict`-insertion order (preserved by unique_map.values() below)
            # keeps them ranked ahead of plain substring caption/tag matches,
            # which have no real relevance ordering of their own.
            combined = []
            unique_map = {}

            for r in results_semantic + results_caption + results_tags:
                fp = r.get("file_path")
                if not fp:
                    continue
                if fp not in unique_map:
                    unique_map[fp] = r
                else:
                    # merge metadata if needed (prefer captioned version)
                    if r.get("caption") and not unique_map[fp].get("caption"):
                        unique_map[fp] = r

            combined = list(unique_map.values())
            if pq.active and pq.text:
                allowed = {r.get("file_path") for r in exif_rows}
                combined = [r for r in combined if r.get("file_path") in allowed][:limit]
            if not combined:
                # Falsy on purpose: handle()'s `result or "No matching
                # media found."` / `0.85 if result else 0.3` logic relies
                # on search() returning something falsy when nothing
                # matched. Returning a non-empty "not found" sentinel
                # string here (as this used to) made that check always
                # truthy, so every zero-result search was reported back
                # to the orchestrator at 0.85 confidence — indistinguishable
                # from a real match.
                return None
            header = f"Found {len(combined)} photos"
            if pq.active and pq.summary:
                header += f" ({pq.summary})"
            lines = [header + ":"]
            for i, r in enumerate(combined, 1):
                path = r.get("file_path", "?")
                caption = r.get("caption") or "(no caption)"
                raw_tags = r.get("tags")
                try:
                    tags_list = json.loads(raw_tags) if raw_tags else []
                    tags = ", ".join(tags_list)
                # Narrowed from a bare `except:` — raw_tags is a DB column
                # that's expected to hold either JSON or nothing; the only
                # realistic failures are malformed JSON or an unexpected
                # non-string type. A bare except also silently swallows
                # KeyboardInterrupt/SystemExit, which has no business being
                # caught while formatting a search result string.
                except (json.JSONDecodeError, TypeError):
                    tags = raw_tags or ""
                lines.append(f"{i}. {path} — {caption} [{tags}]{self._kind_suffix(r)}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[Iris] Search error: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers for #71 / #72 / #75 / #77
    # ------------------------------------------------------------------

    @staticmethod
    def _kind_suffix(record: dict) -> str:
        """' (video, 0:42)' for videos; empty for everything else."""
        if record.get("file_type") != "video":
            return ""
        length = video_mod.format_duration(record.get("duration_seconds"))
        return f" (video, {length})" if length else " (video)"

    def _describe(self, record: dict) -> str:
        caption = record.get("caption") or "(no caption)"
        line = f"{record.get('file_path', '?')} — {caption}"
        if "distance" in record:
            line += f" [distance {record['distance']}]"
        return line + self._kind_suffix(record)

    def _object_matches(self, query: str, limit: int) -> list:
        try:
            return self.db.search_files_by_objects(query, limit)
        except Exception as e:
            logger.warning(f"[Iris] Object search failed: {e}")
            return []

    def _person_search(self, query: str):
        """A reply for 'photos of <known person>', or None if the query doesn't
        name anyone Iris knows (so ordinary search carries on)."""
        try:
            hit = self.faces.person_in_query(query)
        except Exception as e:
            logger.warning(f"[Iris] Person lookup failed: {e}")
            return None
        if hit is None:
            return None
        person, rest = hit
        return self._person_files_reply(person["id"], rest)

    def _person_files_reply(self, ref, rest: str):
        """Photos containing a person, optionally narrowed to those whose
        caption/tags/objects mention a word from *rest* ("beach"). If nothing
        matches the extra words, all of the person's photos are shown, and
        the reply says so rather than quietly ignoring the words."""
        found = self.faces.find_person_files(ref)
        if found is None:
            return None
        person, files = found
        label = self.faces.label(person)
        if not files:
            return {"response": f"I don't have any photos of {label} yet.",
                    "data": {"person": person, "files": []}, "confidence": 0.4}
        note = ""
        words = [w for w in (rest or "").split() if len(w) >= 3]
        if words:
            def hay(f):
                return " ".join(str(f.get(k) or "") for k in ("caption", "tags", "objects")).lower()
            narrowed = [f for f in files if any(w in hay(f) for w in words)]
            if narrowed:
                files, note = narrowed, f" matching '{rest}'"
            else:
                note = f" (none matched '{rest}', so this is all of them)"
        shown = files[:10]
        lines = [f"Found {len(files)} photo(s) of {label}{note}:"]
        lines += [f"{i}. {self._describe(f)}" for i, f in enumerate(shown, 1)]
        if len(files) > len(shown):
            lines.append(f"…and {len(files) - len(shown)} more.")
        return {"response": "\n".join(lines),
                "data": {"person": person, "files": [f["id"] for f in files]}, "confidence": 0.9}

    def find_similar(self, file_id: int, limit: int = 10) -> dict:
        """Photos that look like an existing one (#71), by comparing its CLIP
        embedding with the rest, nearest first. Never raises."""
        try:
            record = self.db.get_file(file_id)
            if not record:
                return {"matches": [], "error": "I couldn't find that photo."}
            vector = self.vector_index.get_embedding(file_id)
            if vector is None and record.get("file_type") == "image":
                path = Path(record["file_path"])
                if path.exists():
                    vector = self.embedder.embed_image(path)
                    if vector is not None:
                        self.vector_index.upsert(file_id, vector)
            if vector is None:
                return {"matches": [], "error": (
                    "I don't have a semantic index entry for that one: semantic search may be "
                    "unavailable, or it hasn't been indexed yet (ask me to index your photos).")}
            limit = max(1, min(int(limit), 50))
            hits = [h for h in self.vector_index.query(vector, top_k=limit + 1) if h[0] != file_id][:limit]
            matches = []
            for fid, dist in hits:
                rec = self.db.get_file(fid)
                if rec:
                    matches.append(dict(rec, distance=round(float(dist), 3)))
            return {"matches": matches, "file_id": file_id}
        except Exception as e:
            logger.error(f"[Iris] find_similar error: {e}")
            return {"matches": [], "error": "Something went wrong looking for similar photos."}

    def reindex_embeddings(self, limit: int = 200) -> dict:
        """Give every photo and video that has no semantic-search embedding one (#71).

        Embeddings are normally made when a file is analysed, so anything
        analysed while CLIP wasn't installed or the model couldn't load (or
        before semantic search existed) is invisible to it until this runs.
        Works in batches of *limit*; run again for more. Files that can't be
        read (HEIC, unsupported codecs) are counted as failed and retried
        next time. Never raises.
        """
        stats = {"indexed": 0, "failed": 0, "remaining": 0, "unavailable": False}
        try:
            limit = max(1, min(int(limit), 2000))
            if self.embedder.embed_text("test") is None or not self.vector_index.available:
                stats["unavailable"] = True
                return stats
            done = self.vector_index.indexed_ids()
            todo = [f for f in self.db.get_files_by_type(("image", "video"), 1_000_000)
                    if f["id"] not in done]
            batch, stats["remaining"] = todo[:limit], max(0, len(todo) - limit)
            for rec in batch:
                path = Path(rec["file_path"])
                vector = None
                try:
                    if not path.exists():
                        pass
                    elif rec.get("file_type") == "video":
                        import tempfile
                        with tempfile.TemporaryDirectory(dir=self.config.cache_dir) as tmp:
                            frames = video_mod.sample_frames(
                                path, tmp, getattr(self.config, "video_frames", 4))
                            vector = video_mod.video_embedding(self.embedder, frames)
                    else:
                        vector = self.embedder.embed_image(path)
                except Exception as e:
                    logger.warning(f"[Iris] Re-index failed for {path}: {e}")
                if vector is not None and self.vector_index.upsert(rec["id"], vector):
                    stats["indexed"] += 1
                else:
                    stats["failed"] += 1
        except Exception as e:
            logger.error(f"[Iris] reindex_embeddings error: {e}")
        return stats

    def detect_objects(self, file_id=None, seconds=None) -> dict:
        """What common objects are in a saved photo (``file_id``), or in view of
        the camera now (one frame) or over ``seconds`` seconds (#77). A photo's
        result is saved to its ``objects`` field so it becomes searchable; camera
        frames are never saved. Never raises."""
        try:
            reason = self.detector.unavailable_reason
            if reason:
                return {"response": f"I can't run object detection: {reason}.", "confidence": 0.3}
            if file_id is not None:
                rec = self.db.get_file(file_id)
                if not rec:
                    return {"response": "I couldn't find that photo.", "confidence": 0.3}
                if rec.get("file_type") != "image" or not Path(rec["file_path"]).exists():
                    return {"response": "I can only look for objects in photos that are still on disk.",
                            "confidence": 0.3}
                dets = [d for d in self.detector.detect(rec["file_path"])
                        if d.confidence >= getattr(self.config, "detector_confidence", 0.4)]
                self.db.update_file_objects(file_id, detection_mod.objects_json(dets))
                result = detection_mod.analyse_stream(
                    _Fixed(dets), [(0.0, None)], getattr(self.config, "detector_confidence", 0.4))
                return {"response": detection_mod.summarise(result, watched=False),
                        "data": {"file_id": file_id, "objects": result.max_count}, "confidence": 0.85}

            if not getattr(self.config, "camera_enabled", False):
                return {"response": ("The camera is switched off. Turn on iris.camera.enabled in your "
                                     "config if you want me to look through it; frames are analysed "
                                     "and thrown away, never saved."), "confidence": 0.5}
            watched = seconds is not None and seconds > 0
            span = min(float(seconds), detection_mod.MAX_WATCH_SECONDS) if watched else 0.0
            try:
                with self._camera_factory() as cam:
                    result = detection_mod.analyse_stream(
                        self.detector,
                        detection_mod.camera_frames(cam, span),
                        getattr(self.config, "detector_confidence", 0.4),
                    )
            except detection_mod.CameraUnavailable as e:
                return {"response": f"I couldn't use the camera: {e}.", "confidence": 0.3}
            return {"response": detection_mod.summarise(result, watched=watched),
                    "data": {"objects": result.max_count, "frames": result.frames,
                             "events": result.events}, "confidence": 0.85}
        except Exception as e:
            logger.error(f"[Iris] detect_objects error: {e}")
            return {"response": "Something went wrong running object detection.", "confidence": 0.2}

    def analyse(self, limit: int = 10) -> str:
        if not self.analyser:
            return "IrisAnalyser is not available."
        result = self.analyser.run_batch(limit)
        analysed = result.get("analysed", 0)
        errors = result.get("errors", 0)
        return f"Analysed {analysed} photos. {errors} errors."

    def find_duplicates(self) -> list:
        """backlog #73 — a whole-library cleanup scan, not the at-ingest check."""
        try:
            return self.ingestor.duplicate_detector.find_all_duplicate_groups()
        except Exception as e:
            logger.error(f"[Iris] find_duplicates error: {e}")
            return []

    # backlog #79. Tunable distance threshold for the greedy clustering
    # below — cosine distance on CLIP embeddings; lower = tighter/more
    # albums, higher = looser/fewer albums. Chosen conservatively (tight)
    # since an over-eager cluster mixing unrelated photos is a worse
    # outcome than several small, clearly-related albums.
    _ALBUM_DISTANCE_THRESHOLD = 0.25
    _ALBUM_MIN_SIZE = 3

    def organize_into_albums(self) -> list:
        """
        Cluster every embedded photo into albums by visual similarity
        (backlog #79), replacing any previous auto-generated albums —
        this re-clusters from scratch each call rather than incrementally
        updating, since adding one new photo can legitimately change
        which cluster several existing photos best belong to.

        Uses the SAME events/event_files tables Athena's... no, Iris's
        own schema already had (unused until now) rather than inventing
        a parallel "albums" concept.
        """
        try:
            embeddings = self.vector_index.get_all_embeddings()
        except Exception as e:
            logger.error(f"[Iris] organize_into_albums: embedding fetch failed: {e}")
            return []
        if len(embeddings) < self._ALBUM_MIN_SIZE:
            return []

        clusters = self._cluster_by_distance(embeddings, self._ALBUM_DISTANCE_THRESHOLD)
        clusters = [c for c in clusters if len(c) >= self._ALBUM_MIN_SIZE]
        if not clusters:
            return []

        try:
            self.db.clear_all_events()
        except Exception as e:
            logger.error(f"[Iris] organize_into_albums: could not clear old albums: {e}")
            return []

        albums = []
        for i, file_ids in enumerate(clusters, start=1):
            try:
                event_id = self.db.create_event(f"Album {i}")
                self.db.add_files_to_event(event_id, file_ids)
                albums.append({"id": event_id, "name": f"Album {i}", "files": file_ids})
            except Exception as e:
                logger.error(f"[Iris] organize_into_albums: failed to create album {i}: {e}")
        return albums

    @staticmethod
    def _cluster_by_distance(
        embeddings: list, threshold: float
    ) -> list:
        """
        Greedy single-linkage-style clustering: for each not-yet-assigned
        embedding, gather every other not-yet-assigned embedding within
        `threshold` cosine distance. O(n^2) — fine for a personal photo
        library, the same trade-off DuplicateDetector's full-library scan
        makes. A dedicated clustering library (scikit-learn) was
        deliberately not added as a new dependency for this.
        """
        import math

        def cosine_distance(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - (dot / (norm_a * norm_b))

        assigned: set = set()
        clusters: list = []
        for i, (file_id, vector) in enumerate(embeddings):
            if file_id in assigned:
                continue
            cluster = [file_id]
            assigned.add(file_id)
            for other_id, other_vector in embeddings[i + 1:]:
                if other_id in assigned:
                    continue
                if cosine_distance(vector, other_vector) <= threshold:
                    cluster.append(other_id)
                    assigned.add(other_id)
            clusters.append(cluster)
        return clusters

    def describe_change(self, file_id_a: int, file_id_b: int) -> dict:
        """
        "What changed between these two photos" (backlog #76) — one
        vision-LLM call given BOTH images together, not two independent
        single-photo captions diffed after the fact (which would lose any
        direct before/after comparison the model itself could make).
        """
        if self.analyser is None:
            return {"description": "", "error": "vision analysis is not configured"}

        file_a = self.db.get_file(file_id_a)
        file_b = self.db.get_file(file_id_b)
        if not file_a or not file_b:
            return {"description": "", "error": "one or both photos weren't found"}

        try:
            path_a = Path(file_a["file_path"])
            path_b = Path(file_b["file_path"])
            if not path_a.exists() or not path_b.exists():
                return {"description": "", "error": "one or both photos are missing on disk"}
            image_a, _ = _load_image_base64(path_a)
            image_b, _ = _load_image_base64(path_b)
        except Exception as e:
            logger.error(f"[Iris] describe_change: image load failed: {e}")
            return {"description": "", "error": "couldn't load those images"}

        prompt = (
            "These are two photos, the first labelled BEFORE and the second "
            "AFTER. Describe what changed between them — what's different, "
            "what's new, what's missing. Be specific and concise."
        )
        description = self.analyser._send_to_ollama([image_a, image_b], prompt)
        return {
            "description": description.strip(),
            "file_id_a": file_id_a,
            "file_id_b": file_id_b,
        }

    def stats(self) -> dict:
        try:
            return {
                "total_files": self.db.file_count(),
                "processed": self.db.processed_count(),
                "pending": self.db.pending_count(),
            }
        except Exception as e:
            logger.error(f"[Iris] Stats error: {e}")
            return {"total_files": 0, "processed": 0, "pending": 0}

    def status(self) -> str:
        try:
            s = self.stats()
            return f"Iris has {s['total_files']} photos indexed, {s['processed']} analysed, {s['pending']} pending."
        except Exception as e:
            logger.error(f"[Iris] Status error: {e}")
            return "Iris status unavailable."


class _Fixed:
    """A detector that returns a fixed list: lets a photo's detections go
    through the same summary code as a camera stream."""

    def __init__(self, detections):
        self._d = detections

    def detect(self, _frame):
        return self._d


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
