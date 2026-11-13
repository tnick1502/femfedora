"""Fixed-endpoint provider adapters. Never expose raw HTTP errors, ASR or demographic data."""

import asyncio
import math
from collections import defaultdict

import httpx

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.core.http import json_request

EMOTIONS = frozenset(("sad", "angry", "happy", "neutral"))


class Deepgram:
    def __init__(self, settings, transport=None):
        self.key, self.transport = settings.deepgram_api_key, transport

    async def transcribe(self, recording):
        async with httpx.AsyncClient(timeout=45, transport=self.transport) as client:
            data = await json_request(
                client,
                "POST",
                "https://api.deepgram.com/v1/listen",
                headers={"Authorization": "Token " + self.key, "Content-Type": "audio/wav"},
                params={
                    "model": "nova-3",
                    "language": "ru",
                    "smart_format": "true",
                    "mip_opt_out": "true",
                    "redact_usage": "true",
                },
                content=recording.wav(),
            )
        try:
            text = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            raise ProviderUnavailable("invalid_transcription") from None
        if not isinstance(text, str):
            raise ProviderUnavailable("invalid_transcription")
        return text.strip()


def aggregate_emotions(results, seconds):
    """Duration-weighted posteriors, single speaker only; no symptoms or clinical scores."""
    model_version = None
    if isinstance(results, dict):
        model_version = results.get("modelVersion")
        results = results.get("results")
    if not isinstance(results, list) or len(results) > 10000:
        raise ProviderUnavailable("invalid_voice_analysis")
    speakers = {
        row.get("finalLabel")
        for row in results
        if isinstance(row, dict)
        and row.get("task") == "diarization"
        and isinstance(row.get("finalLabel"), str)
    }
    if len(speakers) != 1:
        return {"status": "multiple_speakers" if speakers else "insufficient_speech"}
    totals, covered, intervals = defaultdict(float), 0.0, []
    for row in results:
        if not isinstance(row, dict) or row.get("task") != "emotion":
            continue
        try:
            start, end = float(row["startTime"]), float(row["endTime"])
            if not (
                math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= seconds + 0.5
            ):
                continue
            if end - start < 2 or any(start < b and end > a for a, b in intervals):
                continue
            scores = {}
            for prediction in row["prediction"]:
                label, value = prediction["label"], float(prediction["posterior"])
                if label in EMOTIONS and math.isfinite(value) and 0 <= value <= 1:
                    scores[label] = value
            if set(scores) != EMOTIONS or not 0.9 <= sum(scores.values()) <= 1.1:
                continue
            weight = end - start
            for label, value in scores.items():
                totals[label] += weight * value / sum(scores.values())
            covered += weight
            intervals.append((start, end))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not covered:
        return {"status": "insufficient_speech"}
    return {
        "status": "available",
        "source": "behavioral_signals",
        "model": model_version[:80] if isinstance(model_version, str) else "unspecified",
        "emotion": {k: round(totals[k] / covered, 3) for k in sorted(EMOTIONS)},
        "speech_seconds": round(covered, 1),
    }


class BehavioralSignals:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport

    async def analyze(self, recording):
        s = self.settings
        if not (s.behavioral_signals_api_key and s.behavioral_signals_client_id):
            return {"status": "not_configured"}
        if recording.seconds < 2:
            return {"status": "insufficient_speech"}
        from urllib.parse import quote

        base = (
            "https://api.behavioralsignals.com/v5/clients/"
            + quote(s.behavioral_signals_client_id, safe="")
            + "/processes"
        )
        async with httpx.AsyncClient(
            headers={"X-Auth-Token": s.behavioral_signals_api_key},
            timeout=15,
            transport=self.transport,
        ) as client:
            process = await json_request(
                client,
                "POST",
                base + "/audio",
                files={"file": ("recording.wav", recording.wav(), "audio/wav")},
                data={"name": "voice-message", "embeddings": "false"},
            )
            pid = process.get("pid") if isinstance(process, dict) else None
            if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
                raise ProviderUnavailable("invalid_audio_process")
            while True:
                state = await json_request(client, "GET", f"{base}/{pid}")
                status = state.get("status") if isinstance(state, dict) else None
                if status == 2:
                    result = await json_request(client, "GET", f"{base}/{pid}/results")
                    return aggregate_emotions(result, recording.seconds)
                if status not in (0, 1):
                    raise ProviderUnavailable("audio_process_failed")
                await asyncio.sleep(1)
