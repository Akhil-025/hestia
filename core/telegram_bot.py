"""
core/telegram_bot.py

HestiaTelegramBot: the Telegram front-end for the same pipeline the desktop
voice mode and the web UI use (``process_fn`` is ``Hestia.process_text``).

Backlog section 18 (#191-#196)
------------------------------
#191  Inline buttons. A reply that is waiting on a yes/no gets Confirm /
      Cancel buttons, and a fired reminder pushed to Telegram gets Snooze /
      Done buttons. Button presses are routed through the same code the typed
      equivalents use ("yes", "no", "snooze for 10 minutes").
#192  A photo is handed to Iris, a PDF (or an image sent "as a file") to Athena.
      The bot only downloads the file; what to do with it is a hook
      (``ingest_photo_fn`` / ``ingest_document_fn``) wired up in main.py.
#193  ``/help`` is generated from ``INTENT_MODULE_MAP`` (and filtered by the
      chat's role), so it cannot drift from what Hestia can route.
#194  Per-chat roles. ``allowed_chat_ids`` still decides *who may talk to the
      bot*; ``roles`` + ``role_policies`` decide *what each of them may ask for*.
      Chats with no role are "owner" (unrestricted), so existing configs behave
      exactly as before.
#195  A typing indicator is refreshed for as long as a request is running, and
      the request runs off the event loop so the bot stays responsive.
#196  Voice notes (OGG -> ffmpeg -> STT -> the normal pipeline).

Security notes
--------------
* Role checks **fail closed**: a restricted chat whose message can't be
  classified, or that classifies to an unregistered intent, is refused.
* A pending yes/no confirmation belongs to the chat that triggered it. A
  restricted chat can never answer one that came from somewhere else, and one
  chat can't answer another chat's.
* Native location shares overwrite the owner's device location, so restricted
  roles can't send them.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

OWNER_ROLE = "owner"

# Telegram's hard limit for one message, and the Bot API's download limit.
_MAX_MESSAGE_CHARS = 4096
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

# Callback-data prefixes (Telegram caps callback_data at 64 bytes).
_CB_CONFIRM = "cf"   # cf:yes / cf:no
_CB_SNOOZE = "sz"    # sz:<minutes>
_CB_DONE = "ok"      # ok:0

_DENIED = "Sorry, that isn't something you're set up to do here."
_FOREIGN_CONFIRMATION = "That confirmation isn't yours to answer."

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"})

# "yes"/"no" style replies, used only to decide whether a message is an
# answer to a pending confirmation. The orchestrator makes the real decision.
_YES_NO_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|ok|okay|confirm|do it|go ahead|no|nope|nah|cancel|stop|abort)\b",
    re.IGNORECASE,
)
# Confirmation prompts the modules actually emit ("Say yes to send it.").
_CONFIRM_PROMPT_RE = re.compile(r"\bsay yes\b|\(yes/no\)|\byes or no\b|\bare you sure\b", re.IGNORECASE)
# Chronos announces a fired reminder as "Reminder: <text>", optionally after
# "You've arrived at <place>. ".
_REMINDER_RE = re.compile(r"(?:^|\.\s)Reminder:\s", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Roles (#194)
# ---------------------------------------------------------------------------

def _lower_set(values: Any) -> frozenset[str]:
    if not values:
        return frozenset()
    if isinstance(values, str):
        values = [values]
    return frozenset(str(v).strip().lower() for v in values if str(v).strip())


@dataclass(frozen=True)
class RolePolicy:
    """What a role may ask for. ``allow_modules=None`` means any module."""

    allow_modules: Optional[frozenset[str]] = None
    deny_modules: frozenset[str] = field(default_factory=frozenset)
    deny_intents: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "RolePolicy":
        cfg = cfg or {}
        allow = _lower_set(cfg.get("allow_modules"))
        return cls(
            allow_modules=None if (not allow or "*" in allow) else allow,
            deny_modules=_lower_set(cfg.get("deny_modules")),
            deny_intents=_lower_set(cfg.get("deny_intents")),
        )

    def allows_module(self, module: str) -> bool:
        module = (module or "").lower()
        if module in self.deny_modules:
            return False
        return self.allow_modules is None or module in self.allow_modules

    def allows(self, intent: str, module: Optional[str]) -> bool:
        if (intent or "").lower() in self.deny_intents:
            return False
        return module is not None and self.allows_module(module)


def _coerce_chat_id(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _module_for_intent(intent: str) -> Optional[str]:
    """Owning module of *intent*, or None if it isn't registered. A separate
    function so tests can replace it, and so a registry import failure turns
    into a refusal (fail closed) rather than an exception."""
    try:
        from modules.hecate.intent_registry import module_for_intent
        return module_for_intent(intent)
    except Exception:
        logger.exception("[TelegramBot] Could not consult the intent registry.")
        return None


# ---------------------------------------------------------------------------
# /help (#193) and message helpers
# ---------------------------------------------------------------------------

def _humanise(intent: str) -> str:
    try:
        from modules.hecate.intent_registry import strip_module_prefix
        intent = strip_module_prefix(intent)
    except Exception:
        pass
    return intent.replace("_", " ")


def _module_label(module: str) -> str:
    return "General" if module == "core" else module.capitalize()


def build_help_text(
    policy: Optional[RolePolicy] = None,
    module: Optional[str] = None,
    examples_per_module: int = 6,
) -> str:
    """Generate /help from the intent registry. ``policy`` hides what the
    chat's role can't use; ``module`` gives the full list for one module."""
    try:
        from modules.hecate.intent_registry import INTENT_MODULE_MAP
    except Exception:
        logger.exception("[TelegramBot] Could not load the intent registry for /help.")
        return "I can't build the help list right now. Just ask me something in plain English."

    grouped: dict[str, list[str]] = {}
    for intent, owner in INTENT_MODULE_MAP.items():
        if intent == "chat":
            continue
        if policy is not None and not policy.allows(intent, owner):
            continue
        label = _humanise(intent)
        names = grouped.setdefault(owner, [])
        if label not in names:
            names.append(label)

    if module:
        wanted = module.strip().lower()
        if wanted == "general":
            wanted = "core"
        if wanted not in grouped:
            known = ", ".join(sorted(_module_label(m).lower() for m in grouped))
            return f"I don't have a '{module.strip()}' section. Try: {known}."
        body = ", ".join(grouped[wanted])
        text = f"{_module_label(wanted)} - {len(grouped[wanted])} things you can ask:\n{body}"
        return text if len(text) <= _MAX_MESSAGE_CHARS else text[: _MAX_MESSAGE_CHARS - 1] + "…"

    if not grouped:
        return "There's nothing I can help you with from this chat."

    lines = [
        "Just talk to me like you would by voice - text or a voice note. A few things I can do:",
        "",
    ]
    for owner in sorted(grouped, key=lambda m: (m != "core", m)):
        names = grouped[owner]
        shown = ", ".join(names[:examples_per_module])
        extra = len(names) - examples_per_module
        more = f" (+{extra} more)" if extra > 0 else ""
        lines.append(f"{_module_label(owner)}: {shown}{more}")
    lines += ["", "/help <section> lists everything in one section, e.g. /help chronos"]
    text = "\n".join(lines)
    return text if len(text) <= _MAX_MESSAGE_CHARS else text[: _MAX_MESSAGE_CHARS - 1] + "…"


def _split_message(text: str, limit: int = _MAX_MESSAGE_CHARS) -> list[str]:
    """Split *text* into chunks Telegram will accept, preferring line breaks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks


def _safe_filename(name: Optional[str], default: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    base = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return base or default


def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Button layouts (#191). Plain data, so the same layout serves both the
# python-telegram-bot handlers and the raw Bot API push.
# ---------------------------------------------------------------------------

def _confirm_rows() -> list[list[tuple[str, str]]]:
    return [[("✅ Confirm", f"{_CB_CONFIRM}:yes"), ("❌ Cancel", f"{_CB_CONFIRM}:no")]]


def _reminder_rows(snooze_minutes: Iterable[int]) -> list[list[tuple[str, str]]]:
    row = []
    for minutes in snooze_minutes:
        label = f"⏰ {minutes}m" if minutes < 60 else (f"⏰ {minutes // 60}h" if minutes % 60 == 0 else f"⏰ {minutes}m")
        row.append((label, f"{_CB_SNOOZE}:{minutes}"))
    row.append(("✔️ Done", f"{_CB_DONE}:0"))
    return [row]


def _to_markup(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=data) for text, data in row] for row in rows]
    )


def looks_like_reminder(text: str) -> bool:
    return bool(_REMINDER_RE.search(text or ""))


class HestiaTelegramBot:
    def __init__(
        self,
        token: str,
        process_fn: Callable[[str], str],
        allowed_chat_ids: list[int] = None,
        stt=None,
        memory=None,
        *,
        roles: Optional[dict] = None,
        role_policies: Optional[dict] = None,
        classify_fn: Optional[Callable[[str], Union[str, Iterable[str], None]]] = None,
        pending_fn: Optional[Callable[[], bool]] = None,
        snooze_fn: Optional[Callable[[int], str]] = None,
        ingest_photo_fn: Optional[Callable[[str, str], str]] = None,
        ingest_document_fn: Optional[Callable[[str, str], str]] = None,
        should_push_fn: Optional[Callable[[], bool]] = None,
        snooze_minutes: Iterable[int] = (10, 60),
        typing_interval: float = 4.0,
        max_download_bytes: int = _MAX_DOWNLOAD_BYTES,
    ):
        """
        Args:
          token: Telegram bot token from BotFather.
          process_fn: Function to call with user text — returns response string (this is Hestia.process_text).
          allowed_chat_ids: Whitelist of chat IDs. If None or empty (and no ``roles``), allow all (not recommended for production).
          stt: Optional HestiaSTT instance for transcribing voice notes.
          memory: Optional MnemosyneEngine instance. When provided, native
            Telegram location shares (the paperclip → Location attachment,
            not typed text) are persisted via set_device_location() so
            every god can read them back through get_context().
          roles: ``{chat_id: role_name}`` (#194). Chats listed here are
            allowed even if missing from ``allowed_chat_ids``. A chat with no
            entry is ``owner``.
          role_policies: ``{role_name: {allow_modules, deny_modules,
            deny_intents}}``. ``owner`` with no policy is unrestricted; any
            other role without a policy can do nothing (fail closed).
          classify_fn: ``text -> intent | [intents] | None``. Needed to
            enforce a restricted role on free text. Every intent returned
            (use the compound-query splitter) must be allowed.
          pending_fn: ``() -> bool``, whether the orchestrator is waiting on
            a yes/no. Lets Confirm/Cancel buttons appear exactly when they
            should and lets the bot keep confirmations with their chat.
          snooze_fn: ``minutes -> reply text``; run by the Snooze buttons.
            Falls back to sending "snooze for N minutes" through process_fn.
          ingest_photo_fn / ingest_document_fn: ``(local_path, filename) ->
            reply text`` (#192). Photos go to Iris, PDFs to Athena; main.py
            supplies them. The file is deleted after the hook returns.
          should_push_fn: ``() -> bool``; push() is skipped when it says no
            (e.g. while do-not-disturb is on).
          snooze_minutes: Snooze button lengths on a reminder push.
          typing_interval: Seconds between "typing…" refreshes (Telegram
            clears the indicator after ~5 s).
          max_download_bytes: Files above this aren't downloaded (the Bot
            API refuses downloads over 20 MB anyway).
        """
        self.token = token
        self.process_fn = process_fn
        self.allowed_chat_ids = allowed_chat_ids
        self.stt = stt
        self.memory = memory
        self.classify_fn = classify_fn
        self.pending_fn = pending_fn
        self.snooze_fn = snooze_fn
        self.ingest_photo_fn = ingest_photo_fn
        self.ingest_document_fn = ingest_document_fn
        self.should_push_fn = should_push_fn
        self.snooze_minutes = tuple(int(m) for m in snooze_minutes) or (10,)
        self.typing_interval = typing_interval
        self.max_download_bytes = max_download_bytes

        self.roles: dict[int, str] = {}
        for raw_id, role in (roles or {}).items():
            chat_id = _coerce_chat_id(raw_id)
            if chat_id is None:
                logger.warning("[TelegramBot] Ignoring role entry with non-numeric chat id %r.", raw_id)
                continue
            self.roles[chat_id] = str(role).strip().lower() or OWNER_ROLE
        self.role_policies: dict[str, RolePolicy] = {
            str(name).strip().lower(): RolePolicy.from_config(cfg)
            for name, cfg in (role_policies or {}).items()
        }
        for chat_id, role in self.roles.items():
            if role != OWNER_ROLE and role not in self.role_policies:
                logger.warning(
                    "[TelegramBot] Chat %s has role %r but no policy for it; that chat will be refused everything.",
                    chat_id, role,
                )

        self._thread: Optional[threading.Thread] = None
        self._process_lock = threading.Lock()   # Hestia.process_text isn't built for concurrent callers
        self._pending_chat_id: Optional[int] = None
        self._app = ApplicationBuilder().token(token).build()
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        self._app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
        self._app.add_handler(MessageHandler(filters.LOCATION, self._handle_location))
        self._app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))

    def start(self) -> None:
        """Start the bot in a background daemon thread using run_polling."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="TelegramBot")
        self._thread.start()

    def stop(self) -> None:
        """Request bot shutdown."""
        if self._app.running:
            self._app.stop_running()

    def _run(self) -> None:
        """Run the bot event loop — called inside the daemon thread."""
        import asyncio
        asyncio.run(self._app.run_polling(drop_pending_updates=True))

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _is_allowed(self, chat_id: int) -> bool:
        """Return True if chat_id is in the allowlist (or has a role), or if no allowlist is set."""
        known = set(self.allowed_chat_ids or []) | set(self.roles)
        if not known:
            return True
        return chat_id in known

    def role_for(self, chat_id: int) -> str:
        return self.roles.get(chat_id, OWNER_ROLE)

    def _policy_for(self, chat_id: int) -> Optional[RolePolicy]:
        """None means unrestricted. An unknown non-owner role gets an
        empty allow-list, i.e. is refused everything."""
        role = self.role_for(chat_id)
        policy = self.role_policies.get(role)
        if policy is not None:
            return policy
        if role == OWNER_ROLE:
            return None
        return RolePolicy(allow_modules=frozenset())

    def _check_access(
        self, chat_id: int, text: Optional[str] = None, module: Optional[str] = None
    ) -> Optional[str]:
        """Return a refusal message, or None if the chat may proceed.

        ``module`` is for non-text requests (a photo is Iris, a PDF is
        Athena); ``text`` is classified with ``classify_fn``. Runs in a
        worker thread because classification can call the LLM."""
        policy = self._policy_for(chat_id)
        if policy is None:
            return None
        if module is not None:
            return None if policy.allows_module(module) else _DENIED

        if self.classify_fn is None:
            logger.warning("[TelegramBot] Chat %s has a restricted role but no classify_fn; refusing.", chat_id)
            return _DENIED
        try:
            result = self.classify_fn(text or "")
        except Exception:
            logger.exception("[TelegramBot] Classification failed for a restricted chat; refusing.")
            return _DENIED
        intents = [result] if isinstance(result, str) else list(result or [])
        if not intents:
            return _DENIED
        for intent in intents:
            owner = _module_for_intent(intent)
            if owner is None or not policy.allows(intent, owner):
                logger.info("[TelegramBot] Refused intent %r for chat %s (role %s).", intent, chat_id, self.role_for(chat_id))
                return _DENIED
        return None

    def _pending_now(self) -> bool:
        if self.pending_fn is None:
            return False
        try:
            return bool(self.pending_fn())
        except Exception:
            logger.exception("[TelegramBot] pending_fn failed.")
            return False

    def _foreign_pending(self, chat_id: int) -> bool:
        """True if a confirmation is pending that this chat didn't trigger and
        may not answer."""
        if not self._pending_now():
            return False
        owner = self._pending_chat_id
        if owner == chat_id:
            return False
        if owner is None:
            # Raised outside Telegram (desktop/voice): the owner may answer
            # it from here, a restricted role may not.
            return self._policy_for(chat_id) is not None
        return True

    # ------------------------------------------------------------------
    # Running work (#195)
    # ------------------------------------------------------------------

    def _process_locked(self, chat_id: int, text: str) -> tuple[str, bool]:
        """Run process_fn and report whether it left a confirmation pending."""
        with self._process_lock:
            before = self._pending_now()
            response = self.process_fn(text)
            after = self._pending_now()
            created = bool(after and not before)
            if created:
                self._pending_chat_id = chat_id
            elif not after:
                self._pending_chat_id = None
            return response, created

    def _gate_and_process(self, chat_id: int, text: str) -> tuple[Optional[str], str, bool]:
        denial = self._check_access(chat_id, text=text)
        if denial:
            return denial, "", False
        if _YES_NO_RE.match(text) and self._foreign_pending(chat_id):
            return _FOREIGN_CONFIRMATION, "", False
        response, created = self._process_locked(chat_id, text)
        return None, response, created

    async def _run_with_typing(self, chat, fn: Callable, *args):
        """Run blocking *fn* in a worker thread, refreshing "typing…" until it returns."""
        stop = asyncio.Event()

        async def _pulse() -> None:
            while not stop.is_set():
                try:
                    await chat.send_action(ChatAction.TYPING)
                except Exception:
                    logger.debug("[TelegramBot] Could not send typing indicator.", exc_info=True)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.typing_interval)
                except asyncio.TimeoutError:
                    pass

        pulse = asyncio.ensure_future(_pulse())
        try:
            return await asyncio.to_thread(fn, *args)
        finally:
            stop.set()
            await pulse

    def _wants_confirm_buttons(self, text: str, created_pending: bool) -> bool:
        if self.pending_fn is not None:
            return created_pending
        return bool(_CONFIRM_PROMPT_RE.search(text or ""))

    async def _reply(self, message, text: Optional[str], *, confirm: bool = False) -> None:
        """Send *text*, split to fit Telegram, with Confirm/Cancel on the last part."""
        if not text:
            return
        markup = _to_markup(_confirm_rows()) if confirm else None
        chunks = _split_message(text)
        for index, chunk in enumerate(chunks):
            if markup is not None and index == len(chunks) - 1:
                await message.reply_text(chunk, reply_markup=markup)
            else:
                await message.reply_text(chunk)

    async def _answer(self, message, chat, chat_id: int, text: str) -> None:
        """The shared text path: role check, confirmation ownership, run, reply."""
        denial, response, created = await self._run_with_typing(chat, self._gate_and_process, chat_id, text)
        if denial:
            await message.reply_text(denial)
            return
        await self._reply(message, response, confirm=self._wants_confirm_buttons(response, created))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        chat_id = update.effective_chat.id
        logger.info("[TelegramBot] /start from chat_id: %s", chat_id)
        if not self._is_allowed(chat_id):
            await update.message.reply_text("Unauthorised.")
            return
        await update.message.reply_text(
            "Hey! I'm Hestia. Talk to me just like you would by voice. "
            "Send text or a voice note and I'll respond. Send /help to see what I can do."
        )

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help [section] — generated from the intent registry (#193)."""
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            await update.message.reply_text("Unauthorised.")
            return
        args = getattr(context, "args", None)
        section = " ".join(args) if isinstance(args, (list, tuple)) and args else None
        await update.message.reply_text(build_help_text(self._policy_for(chat_id), module=section))

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text message — run process_fn and reply."""
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        user_text = update.message.text.strip()
        if not user_text:
            return
        await self._answer(update.message, update.effective_chat, chat_id, user_text)

    async def _handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle a Telegram location share (paperclip → Location), including
        live-location pings.

        filters.LOCATION matches against update.effective_message, which
        resolves to update.edited_message (not update.message) for live
        location updates after the first ping — update.message is None in
        that case, so reading it directly crashes every single ping. Use
        effective_message, which covers both.
        """
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        # A share overwrites the owner's device location; restricted roles can't.
        if self._policy_for(chat_id) is not None:
            return

        msg = update.effective_message
        if msg is None or msg.location is None:
            return

        if self.memory is None:
            await msg.reply_text("Got your location, but I'm not able to save it right now.")
            return

        loc = msg.location
        try:
            self.memory.set_device_location(loc.latitude, loc.longitude, source="telegram")
            # Live-location pings arrive every ~15-30s; a reply per ping
            # would spam the chat, so only confirm the initial share.
            if update.message is not None:
                await msg.reply_text("Got it — location saved.")
        except Exception:
            logger.error("[TelegramBot] Failed to save location.", exc_info=True)
            if update.message is not None:
                await msg.reply_text("I couldn't save that location.")

    # -- files (#192) -----------------------------------------------------

    async def _download(self, tg_file, directory: str, filename: str) -> str:
        path = os.path.join(directory, filename)
        await tg_file.download_to_drive(path)
        return path

    async def _ingest(self, update: Update, *, module: str, hook, tg_file_getter, filename: str, kind: str) -> None:
        """Download one attachment into a private temp dir, run *hook* on it, reply, clean up."""
        chat_id = update.effective_chat.id
        message = update.effective_message
        if hook is None:
            await message.reply_text(f"I can't file {kind}s in this setup.")
            return
        denial = await asyncio.to_thread(self._check_access, chat_id, None, module)
        if denial:
            await message.reply_text(denial)
            return

        workdir = tempfile.mkdtemp(prefix="hestia_tg_")
        try:
            tg_file = await tg_file_getter()
            path = await self._download(tg_file, workdir, filename)
            reply = await self._run_with_typing(update.effective_chat, hook, path, filename)
            await self._reply(message, reply or f"Got that {kind}.")
        except Exception:
            logger.error("[TelegramBot] Failed to ingest a %s.", kind, exc_info=True)
            await message.reply_text(f"Something went wrong filing that {kind}.")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """A photo → Iris. Telegram sends several sizes; the last is the largest."""
        if not self._is_allowed(update.effective_chat.id):
            return
        message = update.effective_message
        photos = getattr(message, "photo", None)
        if not photos:
            return
        largest = photos[-1]
        size = getattr(largest, "file_size", None)
        if isinstance(size, int) and size > self.max_download_bytes:
            await message.reply_text("That photo is too large for me to download (20 MB limit).")
            return
        unique = getattr(largest, "file_unique_id", None) or "photo"
        await self._ingest(
            update, module="iris", hook=self.ingest_photo_fn, tg_file_getter=largest.get_file,
            filename=_safe_filename(f"telegram_{unique}.jpg", "telegram_photo.jpg"), kind="photo",
        )

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """A PDF → Athena; an image sent "as a file" → Iris; anything else is declined."""
        if not self._is_allowed(update.effective_chat.id):
            return
        message = update.effective_message
        doc = getattr(message, "document", None)
        if doc is None:
            return
        name = _safe_filename(getattr(doc, "file_name", None), "telegram_file")
        mime = (getattr(doc, "mime_type", None) or "").lower()
        ext = Path(name).suffix.lower()

        if ext == ".pdf" or mime == "application/pdf":
            module, hook, kind = "athena", self.ingest_document_fn, "PDF"
            if ext != ".pdf":
                name += ".pdf"
        elif mime.startswith("image/") or ext in _IMAGE_EXTS:
            module, hook, kind = "iris", self.ingest_photo_fn, "image"
        else:
            await message.reply_text("I can only file photos, images and PDFs.")
            return

        size = getattr(doc, "file_size", None)
        if isinstance(size, int) and size > self.max_download_bytes:
            await message.reply_text("That file is too large for me to download (20 MB limit).")
            return
        await self._ingest(
            update, module=module, hook=hook, tg_file_getter=doc.get_file, filename=name, kind=kind,
        )

    # -- voice (#196) -----------------------------------------------------

    def _transcribe_ogg(self, ogg_path: str, wav_path: str) -> str:
        """OGG → 16 kHz mono WAV (ffmpeg) → float32 samples → STT. Blocking."""
        import numpy as np

        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return self.stt._transcribe(audio)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice note — download OGG, transcribe via STT if available, process as text."""
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        if self.stt is None:
            await update.message.reply_text("Voice notes aren't supported in this mode.")
            return

        tmp_path: Optional[str] = None
        wav_path: Optional[str] = None
        try:
            voice_file = await update.message.voice.get_file()
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_path = tmp.name
            wav_path = os.path.splitext(tmp_path)[0] + ".wav"
            await voice_file.download_to_drive(tmp_path)
            text = await self._run_with_typing(update.effective_chat, self._transcribe_ogg, tmp_path, wav_path)
        except subprocess.CalledProcessError:
            await update.message.reply_text("I couldn't process that audio file. Is ffmpeg installed?")
            return
        except Exception as e:
            logger.error("[TelegramBot] Voice handling error: %s", e)
            await update.message.reply_text("Something went wrong processing that voice note.")
            return
        finally:
            _safe_unlink(tmp_path)
            _safe_unlink(wav_path)

        if not text or len(text.strip()) < 2:
            await update.message.reply_text("I couldn't make out that voice note.")
            return

        text = text.strip()
        await update.message.reply_text(f"Heard: {text}")
        try:
            await self._answer(update.message, update.effective_chat, chat_id, text)
        except Exception as e:
            logger.error("[TelegramBot] Voice handling error: %s", e)
            await update.message.reply_text("Something went wrong processing that voice note.")

    # -- inline buttons (#191) ---------------------------------------------

    async def _clear_buttons(self, query) -> None:
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            logger.debug("[TelegramBot] Could not clear inline buttons.", exc_info=True)

    def _snooze(self, minutes: int) -> str:
        with self._process_lock:
            if self.snooze_fn is not None:
                return self.snooze_fn(minutes)
            return self.process_fn(f"snooze for {minutes} minutes")

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """A tap on an inline button."""
        query = update.callback_query
        if query is None:
            return
        try:
            await query.answer()   # stop the button's loading spinner
        except Exception:
            logger.debug("[TelegramBot] Could not answer callback query.", exc_info=True)

        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        kind, _, arg = (query.data or "").partition(":")
        message = query.message
        if message is None:
            return

        if kind == _CB_DONE:
            await self._clear_buttons(query)
            return

        if kind == _CB_CONFIRM and arg in ("yes", "no"):
            if self._foreign_pending(chat_id):
                await message.reply_text(_FOREIGN_CONFIRMATION)
                return
            await self._clear_buttons(query)
            response, created = await self._run_with_typing(
                update.effective_chat, self._process_locked, chat_id, arg
            )
            await self._reply(message, response, confirm=self._wants_confirm_buttons(response, created))
            return

        if kind == _CB_SNOOZE:
            try:
                minutes = int(arg)
            except ValueError:
                return
            denial = await asyncio.to_thread(self._check_access, chat_id, None, "chronos")
            if denial:
                await message.reply_text(denial)
                return
            await self._clear_buttons(query)
            try:
                response = await self._run_with_typing(update.effective_chat, self._snooze, minutes)
            except Exception:
                logger.error("[TelegramBot] Snooze failed.", exc_info=True)
                await message.reply_text("I couldn't snooze that reminder.")
                return
            await self._reply(message, response)

    # ------------------------------------------------------------------
    # Proactive push (#191)
    # ------------------------------------------------------------------

    def notify_chat_ids(self) -> list[int]:
        """Chats proactive messages go to: unrestricted (owner) chats on the allowlist."""
        ids = list(dict.fromkeys(list(self.allowed_chat_ids or []) + list(self.roles)))
        return [c for c in ids if self._policy_for(c) is None]

    def push(self, text: str, *, chat_ids: Optional[Iterable[int]] = None, reminder: Optional[bool] = None) -> int:
        """Send *text* to Telegram from any thread. Uses the Bot API directly, so
        it doesn't depend on the polling event loop. A reminder gets Snooze/Done
        buttons (detected from the text unless *reminder* is given). Returns how
        many chats it reached."""
        if not text:
            return 0
        if self.should_push_fn is not None:
            try:
                if not self.should_push_fn():
                    return 0
            except Exception:
                logger.exception("[TelegramBot] should_push_fn failed; sending anyway.")
        targets = list(chat_ids) if chat_ids is not None else self.notify_chat_ids()
        if not targets:
            return 0

        import requests

        is_reminder = looks_like_reminder(text) if reminder is None else reminder
        rows = _reminder_rows(self.snooze_minutes) if is_reminder else None
        sent = 0
        for chat_id in targets:
            chunks = _split_message(text)
            for index, chunk in enumerate(chunks):
                payload: dict[str, Any] = {"chat_id": chat_id, "text": chunk}
                if rows is not None and index == len(chunks) - 1:
                    payload["reply_markup"] = {
                        "inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows]
                    }
                try:
                    resp = requests.post(
                        f"https://api.telegram.org/bot{self.token}/sendMessage", json=payload, timeout=10
                    )
                    ok = bool(getattr(resp, "ok", False))
                except Exception:
                    # Never log the URL: it contains the bot token.
                    logger.error("[TelegramBot] Push to chat %s failed.", chat_id)
                    ok = False
                if not ok:
                    break
                if index == len(chunks) - 1:
                    sent += 1
        return sent

    def attach_event_bus(self, bus) -> None:
        """Mirror the assistant's proactive announcements (reminders, nudges, the
        morning brief — everything emitted as a ``speak`` event) to Telegram."""
        def _on_speak(data) -> None:
            try:
                text = (data or {}).get("text", "") if isinstance(data, dict) else str(data or "")
                self.push(text)
            except Exception:
                logger.exception("[TelegramBot] speak mirror failed.")

        bus.on("speak", _on_speak)
