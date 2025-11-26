from pathlib import Path
import re

p = Path('chain/WaveLock.py')
src = p.read_text(encoding='utf-8')

start = src.find('def verify(')
if start == -1:
    raise SystemExit('verify() not found in chain/WaveLock.py')

# find end of current verify() by locating next top-level def (or EOF)
m = re.search(r'^\s*def\s+\w+\(', src[start+1:], flags=re.M)
end = start + 1 + (m.start() if m else len(src)-start-1)

prefix, suffix = src[:start], src[end:]

verify_impl = '''
def verify(self, message: str, signature: str):
    \"\"\"Verify commitment + signature (WLv1 or WLv2).\"\"\"
    # 1) Commitment: recompute and compare after stripping any 'WL*:' prefix
    recomputed = self._curvature_hash(self.psi_star)
    stored = str(self.commitment)
    if ":" in stored:
        stored = stored.split(":", 1)[1]
    if recomputed != stored:
        return False

    # 2) Signature: mirror sign() behavior
    import hashlib, json
    import cupy as cp
    c = str(self.commitment)
    # Legacy path: SIGv1 = sha256(message || psi_bytes)
    is_v1 = (c.startswith("WLv1:") or (len(c) == 64 and all(ch in "0123456789abcdef" for ch in c.lower())))
    if is_v1:
        psi_bytes = cp.asnumpy(self.psi_star.ravel()).tobytes()
        expected = hashlib.sha256(message.encode("utf-8") + psi_bytes).hexdigest()
        return (expected == signature)

    # SIGv2 payload: b"SIGv2\\0" || message || \\0 || headerJSON || \\0 || psi*bytes
    mod = __import__(__name__)
    header = {
        "schema": "WLv2",
        "dtype": "float64",
        "order": "C",
        "bc": "periodic",
        "shape": [int(x) for x in self.psi_star.shape],
        "alpha": float(getattr(mod, "alpha", 1.5)),
        "beta":  float(getattr(mod, "beta", 0.0026)),
        "theta": float(getattr(mod, "theta", 1.0e-5)),
        "epsilon": float(getattr(mod, "epsilon", 1.0e-12)),
        "delta": float(getattr(mod, "delta", 1.0e-12)),
    }
    H = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    psi_bytes = cp.asnumpy(self.psi_star.ravel()).tobytes()
    payload = b"SIGv2\\0" + message.encode("utf-8") + b"\\0" + H + b"\\0" + psi_bytes
    expected = hashlib.sha256(payload).hexdigest()
    return (expected == signature)
'''.lstrip('\\n')

p.write_text(prefix + verify_impl + suffix, encoding='utf-8')
print(' verify() replaced successfully')
