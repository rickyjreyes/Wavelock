"""Guard: no conventional cryptographic primitive may enter wavelock/pde_hash/.

The PDE-native primitive's entire value is that it uses NO SHA/SHAKE/BLAKE/etc.
This test parses every module in the package with the ``ast`` module (so prose
in docstrings that *names* the forbidden primitives does not trip it) and fails
if any forbidden module is imported or any forbidden digest symbol is
referenced as a call/attribute.
"""

import ast
import os

import pytest

PKG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "wavelock", "pde_hash")
)

# Forbidden top-level module names (an import of any is an automatic failure).
FORBIDDEN_MODULES = {
    "hashlib",
    "cryptography",
    "blake3",
    "Crypto",          # PyCryptodome / PyCrypto
    "Cryptodome",
    "nacl",            # PyNaCl
    "hmac",
    "_hashlib",
    "_sha3",
    "_md5",
    "_sha256",
    "_sha512",
    "_blake2",
}

# Forbidden digest/cipher/KDF symbol substrings (checked against call + attribute
# names). Lowercased before comparison.
FORBIDDEN_SYMBOL_TOKENS = (
    "sha1", "sha224", "sha256", "sha384", "sha512", "sha3", "sha_",
    "shake", "blake2", "blake3", "blake_", "md5", "ripemd",
    "hmac", "hkdf", "pbkdf", "scrypt", "argon2",
    "aes", "chacha", "salsa20", "siphash", "keccak",
)


def _python_files():
    for name in sorted(os.listdir(PKG_DIR)):
        if name.endswith(".py"):
            yield os.path.join(PKG_DIR, name)


def _imported_modules(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module.split(".")[0])
                mods.add(node.module)
    return mods


def _referenced_symbols(tree):
    """Collect attribute names and called-name identifiers (not docstrings)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


@pytest.mark.parametrize("path", list(_python_files()), ids=os.path.basename)
def test_no_forbidden_module_imports(path):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    mods = _imported_modules(tree)
    bad = mods & FORBIDDEN_MODULES
    assert not bad, f"{os.path.basename(path)} imports forbidden module(s): {sorted(bad)}"


@pytest.mark.parametrize("path", list(_python_files()), ids=os.path.basename)
def test_no_forbidden_digest_symbols(path):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for sym in _referenced_symbols(tree):
        low = sym.lower()
        for token in FORBIDDEN_SYMBOL_TOKENS:
            assert token not in low, (
                f"{os.path.basename(path)} references forbidden symbol "
                f"'{sym}' (token '{token}')"
            )


def test_package_has_modules():
    # Sanity: the guard is actually scanning files.
    files = list(_python_files())
    assert len(files) >= 7, f"expected the full package, found {files}"
