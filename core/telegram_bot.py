import threading
import logging
import os
import tempfile
from typing import Callable, Optional

logging.basicConfig(level=logging.WARNING)

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

class HestiaTelegramBot:
    def __init__(self, token: str, process_fn: Callable[[str], str], allowed_chat_ids: list[int] = None, stt=None):
        """
        Args:
          token: Telegram bot token from BotFather.
          process_fn: Function to call with user text — returns response string (this is Hestia.process_text).
          allowed_chat_ids: Whitelist of chat IDs. If None or empty, allow all (not recommended for production).
          stt: Optional HestiaSTT instance for transcribing voice notes.
        """
        self.token = token
        self.process_fn = process_fn
        self.allowed_chat_ids = allowed_chat_ids
        self.stt = stt
        self._thread: Optional[threading.Thread] = None
        self._app = ApplicationBuilder().token(token).build()
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        self._app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))

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

    def _is_allowed(self, chat_id: int) -> bool:
        """Return True if chat_id is in the allowlist, or if no allowlist is set."""
        if not self.allowed_chat_ids:
            return True
        return chat_id in self.allowed_chat_ids

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        chat_id = update.effective_chat.id
        print(f"[TELEGRAM] chat_id: {chat_id}")
        if not self._is_allowed(chat_id):
            await update.message.reply_text("Unauthorised.")
            return
        await update.message.reply_text(
            "Hey! I'm Hestia. Talk to me just like you would by voice. "
            "Send text or a voice note and I'll respond."
        )

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text message — run process_fn and reply."""
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        user_text = update.message.text.strip()
        if not user_text:
            return
        response = self.process_fn(user_text)
        if response:
            await update.message.reply_text(response)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice note — download OGG, transcribe via STT if available, process as text."""
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        if self.stt is None:
            await update.message.reply_text("Voice notes aren't supported in this mode.")
            return
        try:
            voice_file = await update.message.voice.get_file()
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_path = tmp.name
            await voice_file.download_to_drive(tmp_path)

            # Convert OGG to WAV using ffmpeg subprocess, then transcribe
            import subprocess
            import numpy as np
            wav_path = tmp_path.replace(".ogg", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            # Load WAV as float32 numpy array
            import wave
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            text = self.stt._transcribe(audio)
            os.unlink(tmp_path)
            os.unlink(wav_path)

            if not text or len(text.strip()) < 2:
                await update.message.reply_text("I couldn't make out that voice note.")
                return

            await update.message.reply_text(f"Heard: {text}")
            response = self.process_fn(text)
            if response:
                await update.message.reply_text(response)

        except subprocess.CalledProcessError:
            await update.message.reply_text("I couldn't process that audio file. Is ffmpeg installed?")
        except Exception as e:
            logging.error(f"[TelegramBot] Voice handling error: {e}")
            await update.message.reply_text("Something went wrong processing that voice note.")
