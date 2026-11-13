"""The exact user-selected INT8 model, run locally on CPU. No runtime downloads."""

import asyncio
import hashlib
import os
import threading
from pathlib import Path

from emo_core.core.exceptions import ProviderUnavailable

FILES = {
    "model.int8.onnx": "784d77aaa7a0899a247e72c71d667792087ee4dec985db1f57961767a5f6014f",
    "tokens.txt": "142de7570b3de5b3035ce111a89c228e80e6085273731d944093ddf24fa539cd",
}
REPOSITORY = "nocmt/GigaAM-v3-E2E-CTC-int8-ONNX"
MAX_SECONDS = 25  # Upstream short-form contract; no implicit sentence-cutting or VAD service.


def verify(directory):
    for name, expected in FILES.items():
        with (Path(directory) / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ProviderUnavailable("gigaam_checksum_mismatch")


class GigaAM:
    def __init__(self, settings):
        self.directory, self.threads = Path(settings.gigaam_model_dir), settings.gigaam_threads
        self.recognizer = None
        self.lock = threading.Lock()
        self.running = None

    def installed(self):
        return all((self.directory / name).is_file() for name in FILES)

    def _transcribe(self, recording):
        # The lock remains held even if the request awaiting to_thread is cancelled.
        with self.lock:
            try:
                import numpy as np
                import sherpa_onnx

                if self.recognizer is None:
                    verify(self.directory)
                    self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                        model=str(self.directory / "model.int8.onnx"),
                        tokens=str(self.directory / "tokens.txt"),
                        num_threads=self.threads,
                        sample_rate=16000,
                        feature_dim=64,
                        decoding_method="greedy_search",
                        provider="cpu",
                    )
                stream = self.recognizer.create_stream()
                stream.accept_waveform(
                    16000, np.frombuffer(recording.pcm, dtype="<i2").astype(np.float32) / 32768
                )
                self.recognizer.decode_stream(stream)
                return stream.result.text.strip()
            except (ImportError, OSError, RuntimeError, ValueError):
                raise ProviderUnavailable("gigaam_unavailable") from None

    async def transcribe(self, recording):
        if recording.seconds > MAX_SECONDS:
            raise ProviderUnavailable("gigaam_recording_too_long")
        if self.running is not None and not self.running.done():
            raise ProviderUnavailable("gigaam_busy")
        # Cancelling the HTTP waiter cannot stop native inference. Keep its task and
        # refuse new jobs until it finishes, rather than accumulating executor threads.
        self.running = asyncio.create_task(asyncio.to_thread(self._transcribe, recording))
        self.running.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return await asyncio.shield(self.running)


def download_model():
    """Explicit operator command; verify downloads before publishing either artifact."""
    import argparse
    import tempfile
    import urllib.request

    parser = argparse.ArgumentParser(description="Download and verify GigaAM v3 E2E CTC INT8")
    parser.add_argument("--directory", default="models/gigaam-v3")
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        for name in FILES:
            url = f"https://huggingface.co/{REPOSITORY}/resolve/main/{name}"
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                (Path(temporary) / name).open("wb") as target,
            ):
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 230 * 1024 * 1024:
                        raise SystemExit("Model download exceeded size limit")
                    target.write(chunk)
        verify(temporary)
        for name in FILES:
            os.replace(Path(temporary) / name, directory / name)
    print("GigaAM files downloaded and SHA-256 verified")
