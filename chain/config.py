# chain/config.py
from __future__ import annotations
from dataclasses import dataclass
import os, json

DEFAULT_TARGET = os.getenv(
    "POW_TARGET",
    "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
)

@dataclass
class Config:
    port: int = int(os.getenv("WAVELOCK_PORT", "9001"))
    require_full_verify: bool = bool(int(os.getenv("WAVELOCK_REQUIRE_FULL_VERIFY", "0")))
    reject_if_unpublished: bool = bool(int(os.getenv("WAVELOCK_REJECT_IF_UNPUBLISHED", "0")))
    pow_target: str = DEFAULT_TARGET
    retarget_window: int = int(os.getenv("RETARGET_WINDOW", "20"))
    seeds: list[str] = None

def load_config(path: str | None = None) -> Config:
    cfg = Config()
    seeds_env = os.getenv("SEEDS", "")
    if seeds_env:
        cfg.seeds = [s.strip() for s in seeds_env.split(",") if s.strip()]
    else:
        cfg.seeds = []
    # optional json file (very lightweight)
    if path and os.path.exists(path):
        data = json.load(open(path))
        for k,v in data.items():
            if hasattr(cfg, k):
                setattr(cfg,k,v)
    return cfg
