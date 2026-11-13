import asyncio
import io
import wave
from dataclasses import dataclass

import anyio
from fastapi import HTTPException


@dataclass(frozen=True)
class Recording:
    pcm: bytes
    rate: int = 16000

    @property
    def seconds(self):
        return len(self.pcm) / (self.rate * 2)

    def wav(self):
        out = io.BytesIO()
        with wave.open(out, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.rate)
            f.writeframes(self.pcm)
        return out.getvalue()


async def normalize(data, max_seconds):
    """Decode a bounded in-memory upload. No filenames, network protocols or retained files."""
    import imageio_ffmpeg

    try:
        process = await asyncio.create_subprocess_exec(
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "pipe",
            "-format_whitelist",
            "wav,matroska,webm,mov,ogg,mp3,flac,aac",
            "-threads",
            "1",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            str(max_seconds + 0.2),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-threads",
            "1",
            "-f",
            "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (OSError, RuntimeError):
        raise HTTPException(503, "Декодирование аудио не настроено") from None
    try:
        async with asyncio.timeout(15):
            pcm, _ = await process.communicate(data)
    except BaseException:
        if process.returncode is None:
            process.kill()
        with anyio.CancelScope(shield=True):
            await process.wait()
        raise
    if process.returncode or not pcm or len(pcm) % 2:
        raise HTTPException(422, "Не удалось прочитать запись. Запишите сообщение заново.")
    recording = Recording(pcm)
    if recording.seconds > max_seconds:
        raise HTTPException(422, f"Запись должна быть не длиннее {max_seconds} секунд")
    if recording.seconds < 0.5:
        raise HTTPException(422, "Запись слишком короткая")
    return recording
