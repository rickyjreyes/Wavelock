"""Node configuration. New nodes require public WaveLock-OTS authentication."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
import json
import os

DEFAULT_TARGET = "0000" + "f" * 60


def _bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("1", "true"):
        return True
    if str(value).lower() in ("0", "false"):
        return False
    raise ValueError(f"expected true/false or 1/0, got {value!r}")

@dataclass
class Config:
    port: int = field(default_factory=lambda: int(os.getenv("WAVELOCK_PORT", "9001")))
    require_ots: bool = field(default_factory=lambda: _bool(os.getenv("WAVELOCK_REQUIRE_OTS", "1")))
    require_full_verify: bool = field(default_factory=lambda: _bool(os.getenv("WAVELOCK_REQUIRE_FULL_VERIFY", "0")))
    reject_if_unpublished: bool = field(default_factory=lambda: _bool(os.getenv("WAVELOCK_REJECT_IF_UNPUBLISHED", "0")))
    pow_target: str = field(default_factory=lambda: os.getenv("POW_TARGET", DEFAULT_TARGET))
    retarget_window: int = field(default_factory=lambda: int(os.getenv("RETARGET_WINDOW", "20")))
    seeds: list[str] = field(default_factory=lambda: [s.strip() for s in os.getenv("SEEDS", "").split(",") if s.strip()])

def load_config(path: str | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict):
            raise ValueError("node configuration must be a JSON object")
        unknown = set(data) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
        for key, value in data.items():
            setattr(cfg, key, value)
    for name in ("require_ots", "require_full_verify", "reject_if_unpublished"):
        setattr(cfg, name, _bool(getattr(cfg, name)))
    if type(cfg.port) is not int or not 1 <= cfg.port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    if (not isinstance(cfg.pow_target, str) or len(cfg.pow_target) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in cfg.pow_target)):
        raise ValueError("pow_target must contain exactly 64 hexadecimal digits")
    if not isinstance(cfg.seeds, list) or not all(isinstance(s, str) for s in cfg.seeds):
        raise ValueError("seeds must be a list of host:port strings")
    return cfg
