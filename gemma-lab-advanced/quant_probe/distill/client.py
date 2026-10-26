#!/usr/bin/env python3
"""Tiny OpenAI-compatible client for the LMStudio-served teacher (gemma-4-26b-a4b-qat).

The teacher is a REASONING model: without reasoning_effort=none it burns the whole token
budget on hidden thinking and returns empty content (and is ~10x slower). We always disable
it — a feed blurb needs no chain-of-thought. stdlib only (no requests).
"""
import json, time, urllib.request, urllib.error

URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "google/gemma-4-26b-a4b-qat"

def chat(messages, temp=0.3, max_tokens=48, model=MODEL, url=URL, retries=4, reasoning="none"):
    body = json.dumps({"model": model, "messages": messages, "temperature": temp,
                       "max_tokens": max_tokens, "reasoning_effort": reasoning}).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=180))
            return (resp["choices"][0]["message"].get("content") or "").strip()
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"teacher call failed after {retries} tries: {last}")
