"""Guard: no conventional cryptographic primitive may enter the PDE-native
curvature-capacity package or its audit suite.

Parses every module with ``ast`` (so prose naming the primitives does not trip
it) and fails on any forbidden import or forbidden digest/cipher symbol used as
a call/attribute/name.
"""

import ast
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PKG_DIRS = [
    os.path.join(ROOT, "wavelock", "curvature_capacity"),
    os.path.join(ROOT, "curvature_audit"),
]

FORBIDDEN_MODULES = {
    "hashlib", "cryptography", "blake3", "Crypto", "Cryptodome", "nacl",
    "hmac", "_hashlib", "_sha3", "_md5", "_sha256", "_sha512", "_blake2",
}
FORBIDDEN_SYMBOL_TOKENS = (
    "sha1", "sha224", "sha256", "sha384", "sha512", "sha3", "shake",
    "blake2", "blake3", "md5", "ripemd", "hmac", "hkdf", "pbkdf", "scrypt",
    "argon2", "aes", "chacha", "salsa20", "siphash", "keccak",
)
# audit modules legitimately reference Design A baselines but must not USE a hash;
# _common imports subprocess for git provenance which is fine.


def _python_files():
    for d in PKG_DIRS:
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                yield os.path.join(d, name)


def _imported(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0]); mods.add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0]); mods.add(node.module)
    return mods


def _symbols(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


@pytest.mark.parametrize("path", list(_python_files()), ids=os.path.basename)
def test_no_forbidden_module_imports(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    bad = _imported(tree) & FORBIDDEN_MODULES
    assert not bad, f"{os.path.basename(path)} imports forbidden: {sorted(bad)}"


@pytest.mark.parametrize("path", list(_python_files()), ids=os.path.basename)
def test_no_forbidden_digest_symbols(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for sym in _symbols(tree):
        low = sym.lower()
        for token in FORBIDDEN_SYMBOL_TOKENS:
            assert token not in low, f"{os.path.basename(path)} references '{sym}'"


def test_guard_scans_both_packages():
    assert len(list(_python_files())) >= 12
