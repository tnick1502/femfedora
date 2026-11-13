"""Bounded JSON responses; never expose upstream bodies or exception text."""

import json

import httpx

from emo_core.core.exceptions import ProviderUnavailable


async def json_request(client, method, url, **kwargs):
    try:
        async with client.stream(method, url, **kwargs) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                if len(data) + len(chunk) > 2 * 1024 * 1024:
                    raise ProviderUnavailable("provider_response_too_large")
                data.extend(chunk)
            return json.loads(data)
    except (httpx.HTTPError, ValueError, TypeError):
        raise ProviderUnavailable("provider_unavailable") from None
