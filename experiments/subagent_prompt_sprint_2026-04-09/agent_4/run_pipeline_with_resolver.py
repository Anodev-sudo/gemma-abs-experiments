#!/usr/bin/env python3
from __future__ import annotations

import runpy
import socket
from pathlib import Path

OPENROUTER_HOSTS = {"openrouter.ai", "api.openrouter.ai"}
OPENROUTER_IPS = [
    "104.18.2.115",
    "104.18.3.115",
    "104.22.48.189",
    "104.22.49.189",
    "172.67.40.104",
]


def install_resolver_override() -> None:
    original_getaddrinfo = socket.getaddrinfo

    def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if host in OPENROUTER_HOSTS:
            results = []
            for ip in OPENROUTER_IPS:
                results.extend(original_getaddrinfo(ip, port, family, type, proto, flags))
            return results
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched_getaddrinfo


def main() -> None:
    install_resolver_override()
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "run_synthetic_summary_pipeline.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
