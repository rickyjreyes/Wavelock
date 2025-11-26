from chain.WaveLock import CurvatureKeyPair
from chain.CurvaChain import CurvaChain
from chain.chain_utils import save_chain, load_chain, visualize_psi, tamper_and_test

# ✅ Generate a keypair and visualize
keypair = CurvatureKeyPair(n=4, seed=123)
visualize_psi(keypair.psi_star)

# ✅ Tamper test
tamper_and_test(keypair)

# ✅ Create a chain and add a block
chain = CurvaChain(difficulty=3)
chain.add_block(["Signed curvature message"])
save_chain(chain, "curva_chain.json")

# ✅ Load it back and verify
loaded_chain = load_chain("curva_chain.json")
print("🧬 Loaded block count:", len(loaded_chain.chain))
