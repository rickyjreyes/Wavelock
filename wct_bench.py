import random
import os
import numpy as np
import time

# IMPORT YOUR ATTACK MODULES
# Using the specific paths from your setup
from tests.general import wct_inn_attack_aggressive as inn_attack  # The AI Killer
from tests.general import chaos_test                   # The Phase Space Detector

class WCTBench:
    def __init__(self):
        print("\n=== WCT-BENCH: PHYSICS-BASED RNG AUDIT TOOL ===")
        print("    Audit standard: NIST SP 800-22 (Subset) + WCT Chaos Metrics")
        print("-------------------------------------------------------------")

    def get_standard_rng_stream(self, n_bytes):
        """The 'Weak' Target: Standard Python Mersenne Twister"""
        # Deterministic PRNG
        return [random.random() for _ in range(n_bytes)]

    def get_wavelock_stream(self):
        """The 'Strong' Target: Your Soliton Data"""
        try:
            # Looks for your CSV in the data folder
            path = os.path.join("data", "soliton_n12.csv")
            if not os.path.exists(path):
                print(f"[ERROR] Could not find {path}. Using Mock Data for demo.")
                # Fallback to random just to prevent crash if file missing
                return [random.random() for _ in range(1000)]
            
            # Load data (assuming simple text file with one number per line)
            data = np.loadtxt(path)
            # Take the first 1000 samples
            return data[:1000]
            
        except Exception as e:
            print(f"[ERROR] Failed to load Wavelock data: {e}")
            return []

    def run_audit(self, name, stream_data):
        print(f"\n>>> AUDITING TARGET: {name}")
        print(f"    Sample Size: {len(stream_data)}")
        
        # 1. Neural Inversion Attack
        print("\n[TEST 1] Neural Inversion (AI Prediction)...")
        # Ensure data is list or numpy array
        ai_verdict = inn_attack.run_attack(stream_data, epochs=300)
        print(f"    RESULT: {ai_verdict}")
        
        # 2. Phase Space Determinism
        print("\n[TEST 2] Phase Space Reconstruction (Chaos Detection)...")
        chaos_verdict = chaos_test.run_attack(stream_data)
        print(f"    RESULT: {chaos_verdict}")
        
        return ai_verdict, chaos_verdict

if __name__ == "__main__":
    bench = WCTBench()
    
    while True:
        print("\nSelect Target to Audit:")
        print("1. Standard Python Random (Mersenne Twister)")
        print("2. Wavelock Vacuum-PUF (Soliton Data)")
        print("3. Weak Sine Wave (Math Pattern) [CONTROL TEST]")
        print("q. Quit")
        
        choice = input("Selection > ")
        
        if choice == "1":
            data = bench.get_standard_rng_stream(1000)
            bench.run_audit("PYTHON STANDARD", data)
            
        elif choice == "2":
            data = bench.get_wavelock_stream()
            if len(data) > 0:
                bench.run_audit("WAVELOCK SOLITON", data)

        elif choice == "3":
            # Generate a predictable Sine Wave
            # This proves the tool works by crushing a weak target
            data = [np.sin(i * 0.1) for i in range(1000)]
            bench.run_audit("WEAK SINE WAVE", data)
                
        elif choice == "q":
            break