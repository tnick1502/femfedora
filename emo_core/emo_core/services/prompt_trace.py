"""Private, compressed request snapshots. No HTTP headers, secrets or model reasoning."""

import hashlib
import json
import zlib

RAW_LIMIT = 8 * 1024 * 1024
COMPRESSED_LIMIT = 2 * 1024 * 1024


class PromptRecorder:
    def __init__(self, system_prompt, mode, *, enabled=True):
        self.system_prompt, self.mode, self.requests = system_prompt, mode, []
        self.enabled = enabled
        self.usage_calls = []

    def record_usage(self, purpose, usage):
        safe = {
            key: value
            for key, value in (usage.items() if isinstance(usage, dict) else [])
            if key in ("prompt_tokens", "completion_tokens", "total_tokens")
            and type(value) is int
            and value >= 0
        }
        self.usage_calls.append({"purpose": purpose, "usage": safe or None})
        if self.enabled and self.requests and self.requests[-1]["purpose"] == purpose:
            self.requests[-1]["usage"] = safe or None

    def usage_totals(self):
        keys = {key for call in self.usage_calls for key in (call["usage"] or {})}
        return {
            key: sum((call["usage"] or {}).get(key, 0) for call in self.usage_calls) for key in keys
        }

    def add(self, purpose, messages):
        if not self.enabled:
            return
        self.requests.append(
            {
                "purpose": purpose,
                "messages": [{"role": row["role"], "content": row["content"]} for row in messages],
            }
        )

    def freeze(self):
        if not self.enabled:
            return {"available": False, "reason": "disabled_by_operator"}, None
        raw = json.dumps(
            {
                "version": 1,
                "mode": self.mode,
                "selected_system_prompt": self.system_prompt,
                "requests": self.requests,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        manifest = {
            "available": False,
            "requests": len(self.requests),
            "raw_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "model_called": bool(self.requests),
        }
        if len(raw) > RAW_LIMIT:
            return {**manifest, "reason": "prompt_snapshot_size_limit"}, None
        packed = zlib.compress(raw, level=6)
        if len(packed) > COMPRESSED_LIMIT:
            return {**manifest, "reason": "prompt_snapshot_size_limit"}, None
        return {**manifest, "available": True, "compressed_bytes": len(packed)}, packed


def read_snapshot(packed, expected_hash):
    if len(packed) > COMPRESSED_LIMIT:
        raise ValueError("Invalid prompt snapshot")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(packed, RAW_LIMIT + 1)
    except zlib.error:
        raise ValueError("Invalid prompt snapshot") from None
    if (
        len(packed) > COMPRESSED_LIMIT
        or len(raw) > RAW_LIMIT
        or not decoder.eof
        or decoder.unused_data
        or hashlib.sha256(raw).hexdigest() != expected_hash
    ):
        raise ValueError("Invalid prompt snapshot")
    return json.loads(raw)
