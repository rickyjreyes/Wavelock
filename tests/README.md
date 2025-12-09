
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> ^C
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_hyper.py  

=== WAVELOCK HYPER-TEST HARNESS (WLv3) ===

1) Deterministic Evolution:
{'max_diff': 0, 'min_diff': 0, 'all_zero': True, 'distribution': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

2) Collision Resistance:
{'collisions': 0, 'collision_rate': 0.0}

3) Symbolic Verifier False Acceptance:
{'false_accepts': 0, 'rate': 0.0}

4) Signature Forgery:
{'forgeries': 0, 'rate': 0.0}

5) Drift Sensitivity:
{'sensitivity_failures': 0, 'fail_rate': 0.0, 'rate_ok': 1.0}

6) Resonance Attack:
{'false_accepts': 0, 'rate': 0.0}

7) PDE Inversion Attack:
{'accepted': 0, 'rate': 0.0}

=== DONE ===

(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_attack_nextgen.py

=== WAVELOCK PHASE 2 ADVERSARIAL ATTACK SUITE (v2-final) ===

1) Gradient Surrogate v2:
{'matched': False}

2) Monte Carlo Annealing Attack:
{'matched': False}

3) Fourier Shell Attack:
{'matched': False}

4) Zeta-Phase Layered Attack:
{'matched': False}

5) Curve-Hash v3 Multi-Round Attack:
{'matched': False}

=== DONE ===

(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_extreme.py       

=== WAVELOCK EXTREME TEST SUITE (no curvature functional) ===

1) Avalanche 10k:
{'flip_attempts': 10000, 'failures': 0, 'failure_rate': 0.0}

2) Drift Accumulation:
{'same_commit': False, 'status': 'PASS'}

3) Precision Attack:
{'float32_match': False, 'float16_match': False}

4) Fourier Attack:
{'false_accepts': 0, 'rate': 0.0}

5) Projection Attack:
{'false_accepts': 0, 'rate': 0.0}

6) Multi-GPU Test:
{'skip': True}

7) Quantization Attack:
{'quantized_match': False}

8) Thermal Attack:
{'thermal_match': False}

9) Recombination Attack:
{'false_accepts': 0, 'rate': 0.0}

10) Gradient Surrogate Attack:
{'matched': False}

=== DONE ===

(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_post_quantum.py

=== WAVELOCK POST-QUANTUM ATTACK SUITE (WLv3) ===

1) Grover-Sim Attack:
{'matched': False}

2) QAOA-Sim Attack:
{'matched': False}

3) QFT-Based Reconstruction:
{'matched': False}

4) Quantum Gibbs Sampling Attack:
{'matched': False}

5) Quantum Random Walk Collision Search:
{'matched': False}

=== DONE ===
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_exotic.py      

=== WAVELOCK PHASE 4 — MAXIMUM ADVERSARIAL SUITE (WLv3) ===

1) QPE Simulation Attack:
{'matched': False}

2) HHL Linear-System Attack:
{'matched': False}

3) Multiscale Wavelet Attack:
{'matched': False}

4) Manifold Learning Attack:
{'matched': False}

5) Adjoint PDE Inversion:
{'matched': False}

6) Dual-Space ψ/∇ψ Attack:
{'matched': False}

=== DONE ===
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_curvature_attack.py
=== WCT CURVATURE ATTACK (n=32, steps=500) ===
  0%|                                                                                                          | 0/500 [00:00<?, ?it/s][CURV] iter=0 loss=574243.42377 main=574243.42377 E=0.00000
 20%|███████████████████▏                                                                            | 100/500 [00:16<01:06,  6.00it/s][CURV] iter=100 loss=574229.19026 main=574229.19026 E=0.00000
 40%|██████████████████████████████████████▍                                                         | 200/500 [00:32<00:47,  6.26it/s][CURV] iter=200 loss=574213.85349 main=574213.85349 E=0.00000
 60%|█████████████████████████████████████████████████████████▌                                      | 300/500 [00:49<00:38,  5.13it/s][CURV] iter=300 loss=574198.83540 main=574198.83540 E=0.00000
 80%|████████████████████████████████████████████████████████████████████████████▊                   | 400/500 [01:12<00:15,  6.31it/s][CURV] iter=400 loss=574184.13604 main=574184.13604 E=0.00000
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:30<00:00,  5.55it/s]

=== RESULT ===
Final loss: 574169.8976291103
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_dual_hamiltonian.py

=== DUAL-HAMILTONIAN PSEUDO-INVERSE ATTACK (n=32, steps=500, T=20) ===
  0%|                                                                                                          | 0/500 [00:00<?, ?it/s][DualHam] iter=   0 loss=917213.769321
 20%|███████████████████▏                                                                            | 100/500 [00:17<01:04,  6.21it/s][DualHam] iter= 100 loss=916425.995430
 40%|██████████████████████████████████████▍                                                         | 200/500 [00:33<00:46,  6.41it/s][DualHam] iter= 200 loss=916425.970029
 60%|█████████████████████████████████████████████████████████▌                                      | 300/500 [00:50<00:37,  5.35it/s][DualHam] iter= 300 loss=916425.970029
 80%|████████████████████████████████████████████████████████████████████████████▊                   | 400/500 [01:08<00:15,  6.29it/s][DualHam] iter= 400 loss=916425.970029
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:24<00:00,  5.93it/s]
Final loss = 916425.9700287454
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_inn.py     
[Epoch 1/20] train_loss=1.1029e+03, test_loss=1.1252e+02
[Epoch 2/20] train_loss=3.1514e+02, test_loss=2.5333e+01
[Epoch 3/20] train_loss=3.1068e+01, test_loss=3.8304e+00
[Epoch 4/20] train_loss=1.5185e+01, test_loss=2.3196e+00
[Epoch 5/20] train_loss=5.1948e+00, test_loss=2.1035e+00
[Epoch 6/20] train_loss=2.5841e+00, test_loss=1.8231e+00
[Epoch 7/20] train_loss=2.0422e+00, test_loss=1.7419e+00
[Epoch 8/20] train_loss=1.6299e+00, test_loss=1.6581e+00
[Epoch 9/20] train_loss=1.4336e+00, test_loss=1.5912e+00
[Epoch 10/20] train_loss=1.4145e+00, test_loss=1.5345e+00
[Epoch 11/20] train_loss=1.6545e+00, test_loss=1.4851e+00
[Epoch 12/20] train_loss=1.8921e+00, test_loss=1.4393e+00
[Epoch 13/20] train_loss=1.5588e+00, test_loss=1.3780e+00
[Epoch 14/20] train_loss=1.1770e+00, test_loss=1.3343e+00
[Epoch 15/20] train_loss=1.3006e+00, test_loss=1.2993e+00
[Epoch 16/20] train_loss=1.5430e+00, test_loss=1.2741e+00
[Epoch 17/20] train_loss=1.6961e+00, test_loss=1.2378e+00
[Epoch 18/20] train_loss=7.6249e-01, test_loss=1.1809e+00
[Epoch 19/20] train_loss=4.9184e-01, test_loss=1.1416e+00
[Epoch 20/20] train_loss=4.4271e-01, test_loss=1.1139e+00

=== Neural Inversion Attack Result ===
If test_loss stays high (O(1) or larger), f_θ fails to learn an inverse → strong one-way evidence.
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_jacob.py

=== TRUE BACKPROP JACOBIAN ATTACK (TBJA) ===

{'matched': False, 'final_loss': 225.21997950223005}

=== DONE ===

(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_lyapunov.py
[Lyapunov-Perron] step=0, total_loss=1.4265e+02, cons=9.5621e+01, end=4.7031e+00
[Lyapunov-Perron] step=100, total_loss=1.2240e+02, cons=7.9156e+01, end=4.3245e+00
[Lyapunov-Perron] step=200, total_loss=1.0564e+02, cons=6.5901e+01, end=3.9738e+00
[Lyapunov-Perron] step=300, total_loss=9.1625e+01, cons=5.5137e+01, end=3.6489e+00
[Lyapunov-Perron] step=400, total_loss=7.9789e+01, cons=4.6311e+01, end=3.3478e+00
[Lyapunov-Perron] step=500, total_loss=6.9708e+01, cons=3.9019e+01, end=3.0689e+00
[Lyapunov-Perron] step=600, total_loss=6.1095e+01, cons=3.2989e+01, end=2.8106e+00
[Lyapunov-Perron] step=700, total_loss=5.3686e+01, cons=2.7972e+01, end=2.5714e+00
[Lyapunov-Perron] step=800, total_loss=4.7292e+01, cons=2.3790e+01, end=2.3501e+00
[Lyapunov-Perron] step=900, total_loss=4.1740e+01, cons=2.0285e+01, end=2.1455e+00
[Lyapunov-Perron] step=1000, total_loss=3.6899e+01, cons=1.7335e+01, end=1.9564e+00
[Lyapunov-Perron] step=1100, total_loss=3.2658e+01, cons=1.4840e+01, end=1.7818e+00
[Lyapunov-Perron] step=1200, total_loss=2.8955e+01, cons=1.2749e+01, end=1.6207e+00
[Lyapunov-Perron] step=1300, total_loss=2.5697e+01, cons=1.0974e+01, end=1.4722e+00
[Lyapunov-Perron] step=1400, total_loss=2.2843e+01, cons=9.4868e+00, end=1.3356e+00
[Lyapunov-Perron] step=1499, total_loss=2.0344e+01, cons=8.2333e+00, end=1.2111e+00

=== Lyapunov–Perron Attack Result ===
Final total loss=20.344438
If this cannot drive end_loss near 0, backward synchronization fails (good for one-wayness).
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_tangent.py 
=== Tangent-Space Collapse Analysis (n=4) ===
Approximate top singular value σ_max ≈ 0.4830706453262713
If σ_max << 1 or many directions appear null, J is strongly contractive / rank-deficient.
=== Tangent-Space Collapse Analysis (n=4) ===
Approximate top singular value σ_max ≈ 0.51085173143275
If σ_max << 1 or many directions appear null, J is strongly contractive / rank-deficient.
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_tbja.py   
=== TBJA-Cluster: 1 GPUs, n=6, steps=2000 ===
Target hash: 1dd384fc82a0ce5b18ef29448fd3f8223da2a54f45c65ac17fdb0245c7296413
[GPU 0] Finished. Best loss=6.985282749399619 at iter 1998

=== TBJA-Cluster Results ===
GPU 0: {'matched': False, 'best_loss': 6.985282749399619, 'best_iter': 1998}
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_claude3.py

=== WAVELOCK V3 ATTACK SUITE ===

1) SVD Low-Rank Attack (v3):
{'broken': False, 'results': [{'rank': 1, 'matched': False, 'hash': '9e1d86101ecac7f2'}, {'rank': 2, 'matched': False, 'hash': '8214bc92b5754128'}, {'rank': 3, 'matched': False, 'hash': '40552a03a4fbcfc6'}, {'rank': 4, 'matched': False, 'hash': '84c72487a105b8d2'}, {'rank': 5, 'matched': False, 'hash': '1f5e49cbf3aa5ae5'}, {'rank': 6, 'matched': False, 'hash': '469549c81acc353e'}, {'rank': 7, 'matched': False, 'hash': 'c8b033b517945c1e'}]}

2) Laplacian Eigen Attack (v3):
{'broken': False, 'results': [{'n_eigs': 1, 'matched': False, 'hash': '2aca68c8878abf74'}, {'n_eigs': 2, 'matched': False, 'hash': '824b58fecd61d502'}, {'n_eigs': 3, 'matched': False, 'hash': 'aab68dded03d1314'}, {'n_eigs': 4, 'matched': False, 'hash': 'a085c6ae071a02e2'}, {'n_eigs': 5, 'matched': False, 'hash': '93fc5551590619c7'}, {'n_eigs': 6, 'matched': False, 'hash': 'af6889cd392eb994'}, {'n_eigs': 7, 'matched': False, 'hash': '0d55a31d89bf7a52'}, {'n_eigs': 8, 'matched': False, 'hash': '398ce7348f3e55e1'}, {'n_eigs': 9, 'matched': False, 'hash': 'a86af49f314442e2'}, {'n_eigs': 10, 'matched': False, 'hash': '84f46bc355498d73'}, {'n_eigs': 11, 'matched': False, 'hash': '93daaddf178193ae'}, {'n_eigs': 12, 'matched': False, 'hash': 'e6c65cb13a1eba60'}, {'n_eigs': 13, 'matched': False, 'hash': 'a01b8deb72fb0c77'}, {'n_eigs': 14, 'matched': False, 'hash': '29a6fc8557b735a5'}, {'n_eigs': 15, 'matched': False, 'hash': '0509fdffbbb398bf'}, {'n_eigs': 16, 'matched': False, 'hash': '5eba6d44a189946f'}, {'n_eigs': 17, 'matched': False, 'hash': 'ba462a14a4dab4d5'}, {'n_eigs': 18, 'matched': False, 'hash': '5a8789bad0b185e7'}, {'n_eigs': 19, 'matched': False, 'hash': '87d3b9c84c582661'}]}

3) Precision Cascade Attack (v3):
{'results': [{'precision': 'float64', 'matched': True, 'hash': '06be56805bfaab54'}, {'precision': 'float32', 'matched': False, 'hash': 'fed5d74114e38e7a'}, {'precision': 'float16', 'matched': False, 'hash': '66798f0722cafada'}, {'quant_bits': 8, 'matched': False, 'hash': '329bf6e48cd7d06f'}, {'quant_bits': 16, 'matched': False, 'hash': '76f7a5fd8de557ba'}, {'quant_bits': 24, 'matched': False, 'hash': '5ad56828bc32e2ce'}, {'quant_bits': 32, 'matched': False, 'hash': '30b76c2b5269b97f'}]}

4) Symmetry Attack (v3):
{'broken': False, 'results': [{'symmetry': 'rot90_1', 'matched': False}, {'symmetry': 'rot90_2', 'matched': False}, {'symmetry': 'rot90_3', 'matched': False}, {'symmetry': 'fliplr', 'matched': False}, {'symmetry': 'flipud', 'matched': False}, {'symmetry': 'transpose', 'matched': False}]}

=== DONE ===




============================
   WAVELOCK TIER-Ω ATTACK
============================


=== EigenmodeBundle (K=8, n=32, steps=200) ===
EigenmodeBundle:   0%|                                                                                         | 0/200 [00:00<?, ?it/s][EigenmodeBundle] iter=   0  L_total=0.474307  L_main=0.044517  E_guess=0.000000  curv_term=4.297899e+02
EigenmodeBundle:  25%|████████████████████                                                            | 50/200 [00:19<01:03,  2.38it/s][EigenmodeBundle] iter=  50  L_total=0.106409  L_main=0.046148  E_guess=434690.536302  curv_term=6.004675e+01
EigenmodeBundle:  50%|███████████████████████████████████████▌                                       | 100/200 [01:08<02:16,  1.36s/it][EigenmodeBundle] iter= 100  L_total=0.091857  L_main=0.047323  E_guess=1306759.913935  curv_term=4.420007e+01
EigenmodeBundle:  75%|███████████████████████████████████████████████████████████▎                   | 150/200 [01:41<00:21,  2.30it/s][EigenmodeBundle] iter= 150  L_total=0.074662  L_main=0.050300  E_guess=7663148.789643  curv_term=2.380895e+01

=== Multi-Resolution Envelope Attack (n=32, steps=150) ===
MultiRes:   0%|                                                                                                | 0/150 [00:00<?, ?it/s][MultiRes] iter=   0  L_total=0.362511  L_main=0.325796  E_guess=2355096.296756
MultiRes:  13%|███████████▌                                                                           | 20/150 [02:43<08:14,  3.81s/it][MultiRes] iter=  20  L_total=0.440226  L_main=0.371323  E_guess=250335.185830
MultiRes:  27%|███████████████████████▏                                                               | 40/150 [03:43<05:15,  2.87s/it][MultiRes] iter=  40  L_total=0.553456  L_main=0.479768  E_guess=188559.590807
MultiRes:  40%|██████████████████████████████████▊                                                    | 60/150 [04:40<04:10,  2.79s/it][MultiRes] iter=  60  L_total=0.593018  L_main=0.539584  E_guess=674334.020267
MultiRes:  53%|██████████████████████████████████████████████▍                                        | 80/150 [05:35<03:12,  2.75s/it][MultiRes] iter=  80  L_total=0.747693  L_main=0.592650  E_guess=3942.146888
MultiRes:  67%|█████████████████████████████████████████████████████████▎                            | 100/150 [06:36<02:25,  2.90s/it][MultiRes] iter= 100  L_total=0.867375  L_main=0.724796  E_guess=6572.141913
MultiRes:  80%|████████████████████████████████████████████████████████████████████▊                 | 120/150 [07:34<01:24,  2.82s/it][MultiRes] iter= 120  L_total=0.863732  L_main=0.748765  E_guess=22227.134969
MultiRes:  93%|████████████████████████████████████████████████████████████████████████████████▎     | 140/150 [08:30<00:28,  2.84s/it][MultiRes] iter= 140  L_total=0.887519  L_main=0.793140  E_guess=60868.252658

=== NLS Surrogate Inverse Attack (n=32, steps=80) ===
NLS:   0%|                                                                                                      | 0/80 [00:00<?, ?it/s][NLS] iter=   0  L_total=0.063909  L_main=0.063207  E_guess=436185935.430978
NLS:  24%|██████████████████████                                                                       | 19/80 [00:00<00:01, 46.16it/s][NLS] iter=  20  L_total=0.063412  L_main=0.063192  E_guess=630723222.644423
NLS:  44%|████████████████████████████████████████▋                                                    | 35/80 [00:00<00:00, 62.50it/s][NLS] iter=  40  L_total=0.076874  L_main=0.063260  E_guess=25181275.714742
NLS:  65%|████████████████████████████████████████████████████████████▍                                | 52/80 [00:00<00:00, 71.14it/s][NLS] iter=  60  L_total=0.067315  L_main=0.063391  E_guess=139073896.859029

=== Topological Mapper Homotopy Attack (n=32, steps=200) ===
TopoMap:   0%|                                                                                                 | 0/200 [00:00<?, ?it/s][TopoMap] iter=   0  current_loss=0.115446  best_loss=0.115446
TopoMap:  10%|████████▎                                                                               | 19/200 [00:00<00:04, 40.17it/s][TopoMap] iter=  20  current_loss=0.055694  best_loss=0.055694
TopoMap:  20%|█████████████████▏                                                                      | 39/200 [00:00<00:03, 40.49it/s][TopoMap] iter=  40  current_loss=0.055556  best_loss=0.055556
TopoMap:  30%|█████████████████████████▉                                                              | 59/200 [00:01<00:03, 44.78it/s][TopoMap] iter=  60  current_loss=0.055528  best_loss=0.055528
TopoMap:  40%|██████████████████████████████████▊                                                     | 79/200 [00:01<00:02, 42.93it/s][TopoMap] iter=  80  current_loss=0.055209  best_loss=0.055209
TopoMap:  50%|███████████████████████████████████████████▌                                            | 99/200 [00:02<00:02, 45.70it/s][TopoMap] iter= 100  current_loss=0.054282  best_loss=0.054282
TopoMap:  60%|███████████████████████████████████████████████████▊                                   | 119/200 [00:02<00:01, 44.89it/s][TopoMap] iter= 120  current_loss=0.053760  best_loss=0.053760
TopoMap:  70%|████████████████████████████████████████████████████████████▍                          | 139/200 [00:03<00:01, 44.36it/s][TopoMap] iter= 140  current_loss=0.053289  best_loss=0.053289
TopoMap:  80%|█████████████████████████████████████████████████████████████████████▏                 | 159/200 [00:03<00:00, 41.54it/s][TopoMap] iter= 160  current_loss=0.053139  best_loss=0.053139
TopoMap:  90%|█████████████████████████████████████████████████████████████████████████████▊         | 179/200 [00:04<00:00, 41.00it/s][TopoMap] iter= 180  current_loss=0.052570  best_loss=0.052570

=== TIER-OMEGA ENSEMBLE SUMMARY ===
Agent: EigenmodeBundle       best_loss=    0.047281  corr(ψ0_est, ψ0_true)=-0.044310
Agent: MultiResEnvelope      best_loss=    0.346536  corr(ψ0_est, ψ0_true)=+0.041304
Agent: NLSsurrogate          best_loss=    0.063199  corr(ψ0_est, ψ0_true)=-0.028223
Agent: TopologicalMapper     best_loss=    0.051407  corr(ψ0_est, ψ0_true)=+0.068649

=== TIER-OMEGA FINAL RESULT ===
Best agent        : EigenmodeBundle
Best ensemble loss: 0.047281
Final correlation : -0.044310
================================



====================================
   WAVELOCK WCT SURROGATE ATTACKS
====================================


=== CURVATURE RESONANCE-MANIFOLD ATTACK ===
CurvRes:   0%|                                                                                                 | 0/200 [00:00<?, ?it/s][CurvRes] iter=   0  L=0.242548  L_main=0.057454  L_curv=1.850940e+02  E_guess=0.000000
CurvRes:  25%|██████████████████████                                                                  | 50/200 [00:52<02:03,  1.22it/s][CurvRes] iter=  50  L=0.060077  L_main=0.056775  L_curv=3.302104e+00  E_guess=4985748.226051
CurvRes:  50%|███████████████████████████████████████████▌                                           | 100/200 [01:33<01:21,  1.22it/s][CurvRes] iter= 100  L=0.059351  L_main=0.058134  L_curv=1.216755e+00  E_guess=2441176.044084
CurvRes:  75%|█████████████████████████████████████████████████████████████████▎                     | 150/200 [02:14<00:40,  1.23it/s][CurvRes] iter= 150  L=0.059792  L_main=0.058548  L_curv=1.244356e+00  E_guess=2471736.747020

=== MULTI-RES CHAOTIC ENVELOPE DESCENT ===
MultiRes:   0%|                                                                                                | 0/150 [00:00<?, ?it/s][MultiRes] iter=   0  L=0.381239  L_main=0.379126  E_guess=189393.626817
MultiRes:  13%|███████████▌                                                                           | 20/150 [02:08<13:38,  6.30s/it][MultiRes] iter=  20  L=0.375673  L_main=0.373980  E_guess=2976015.799773
MultiRes:  27%|███████████████████████▏                                                               | 40/150 [03:58<08:04,  4.40s/it][MultiRes] iter=  40  L=0.397383  L_main=0.387845  E_guess=36917.514212
MultiRes:  40%|██████████████████████████████████▊                                                    | 60/150 [05:25<06:24,  4.27s/it][MultiRes] iter=  60  L=0.395533  L_main=0.393344  E_guess=184478.800287
MultiRes:  53%|██████████████████████████████████████████████▍                                        | 80/150 [07:54<15:12, 13.04s/it][MultiRes] iter=  80  L=0.436930  L_main=0.423969  E_guess=22130.632379
MultiRes:  67%|█████████████████████████████████████████████████████████▎                            | 100/150 [10:57<06:02,  7.25s/it][MultiRes] iter= 100  L=0.443655  L_main=0.424639  E_guess=10343.373539
MultiRes:  80%|████████████████████████████████████████████████████████████████████▊                 | 120/150 [14:03<04:38,  9.28s/it][MultiRes] iter= 120  L=0.431569  L_main=0.416483  E_guess=16660.974828
MultiRes:  93%|████████████████████████████████████████████████████████████████████████████████▎     | 140/150 [16:14<00:48,  4.82s/it][MultiRes] iter= 140  L=0.390471  L_main=0.390441  E_guess=682394.803441

=== EIGENMODE BUNDLE ATTACK ===
EigenBundle:   0%|                                                                                             | 0/200 [00:00<?, ?it/s][EigenBundle] iter=   0  L=0.057454
EigenBundle:  25%|█████████████████████                                                               | 50/200 [00:33<01:19,  1.88it/s][EigenBundle] iter=  50  L=0.057453
EigenBundle:  50%|█████████████████████████████████████████▌                                         | 100/200 [01:12<01:16,  1.30it/s][EigenBundle] iter= 100  L=0.057451
EigenBundle:  75%|██████████████████████████████████████████████████████████████▎                    | 150/200 [01:42<00:25,  1.94it/s][EigenBundle] iter= 150  L=0.057449

=== NLS SURROGATE INVERSE ATTACK ===
NLS:   0%|                                                                                                      | 0/80 [00:00<?, ?it/s][NLS] iter=   0  L=0.080291  L_main=0.070807  E_guess=17621252.943515
NLS:  20%|██████████████████▌                                                                          | 16/80 [00:00<00:01, 50.87it/s][NLS] iter=  20  L=0.085533  L_main=0.070819  E_guess=37536845.474163
NLS:  50%|██████████████████████████████████████████████▌                                              | 40/80 [00:00<00:00, 53.48it/s][NLS] iter=  40  L=0.073010  L_main=0.070825  E_guess=3552876.915554
NLS:  74%|████████████████████████████████████████████████████████████████████▌                        | 59/80 [00:01<00:00, 55.24it/s][NLS] iter=  60  L=0.078326  L_main=0.070861  E_guess=12449055.924593

=== TOPOLOGICAL MAPPER HOMOTOPY ATTACK ===
TopoMap:   0%|                                                                                                 | 0/150 [00:00<?, ?it/s][TopoMap] iter=   0  current=0.075925  best=0.075925
TopoMap:  13%|███████████▏                                                                            | 19/150 [00:01<00:06, 18.73it/s][TopoMap] iter=  20  current=0.069017  best=0.069017
TopoMap:  26%|██████████████████████▉                                                                 | 39/150 [00:02<00:05, 18.82it/s][TopoMap] iter=  40  current=0.064412  best=0.064412
TopoMap:  40%|███████████████████████████████████▏                                                    | 60/150 [00:03<00:04, 18.63it/s][TopoMap] iter=  60  current=0.057891  best=0.057891
TopoMap:  53%|██████████████████████████████████████████████▎                                         | 79/150 [00:04<00:03, 19.23it/s][TopoMap] iter=  80  current=0.052683  best=0.052683
TopoMap:  66%|██████████████████████████████████████████████████████████                              | 99/150 [00:05<00:02, 19.20it/s][TopoMap] iter= 100  current=0.050528  best=0.050528
TopoMap:  80%|█████████████████████████████████████████████████████████████████████▌                 | 120/150 [00:06<00:01, 19.16it/s][TopoMap] iter= 120  current=0.050143  best=0.050143
TopoMap:  93%|████████████████████████████████████████████████████████████████████████████████▌      | 139/150 [00:07<00:00, 19.60it/s][TopoMap] iter= 140  current=0.050033  best=0.050033

=== PHASE-SPACE ADJOINT INJECTION ATTACK ===
PhaseSpace:   0%|                                                                                              | 0/120 [00:00<?, ?it/s][PhaseSpace] iter=   0  L=0.067469  L_main=0.066520  E_guess=2145997.650004
PhaseSpace:  15%|████████████▊                                                                        | 18/120 [00:00<00:03, 28.89it/s][PhaseSpace] iter=  20  L=0.073801  L_main=0.066546  E_guess=11976126.260253
PhaseSpace:  32%|███████████████████████████▋                                                         | 39/120 [00:01<00:02, 27.52it/s][PhaseSpace] iter=  40  L=0.067673  L_main=0.066730  E_guess=2139295.332605
PhaseSpace:  50%|██████████████████████████████████████████▌                                          | 60/120 [00:02<00:02, 27.70it/s][PhaseSpace] iter=  60  L=0.072946  L_main=0.066874  E_guess=9520991.966768
PhaseSpace:  65%|███████████████████████████████████████████████████████▎                             | 78/120 [00:02<00:01, 26.19it/s][PhaseSpace] iter=  80  L=0.067341  L_main=0.066982  E_guess=1474577.753543
PhaseSpace:  82%|██████████████████████████████████████████████████████████████████████▏              | 99/120 [00:03<00:00, 26.54it/s][PhaseSpace] iter= 100  L=0.070312  L_main=0.066947  E_guess=5072666.553271

=== ATTACK SUMMARY ===
CurvatureResonance      loss=  0.056198  corr(ψ0_est, ψ0_true)=+0.021457
MultiResEnvelope        loss=  0.369495  corr(ψ0_est, ψ0_true)=+0.044992
EigenmodeBundle         loss=  0.057448  corr(ψ0_est, ψ0_true)=+0.009986
NLSsurrogate            loss=  0.070985  corr(ψ0_est, ψ0_true)=-0.005278
TopologicalMapper       loss=  0.050033  corr(ψ0_est, ψ0_true)=+0.053583
PhaseSpaceAdjoint       loss=  0.067149  corr(ψ0_est, ψ0_true)=+0.014610

=== FINAL RESULT ===
Best attack       : TopologicalMapper
Best loss         : 0.050033
Best correlation  : +0.053583
====================================

 1) 
(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_inn_glow.py 
[Epoch 1] train_loss=11.3051, test_loss=2.5697
[Epoch 2] train_loss=9.6171, test_loss=2.3389
[Epoch 3] train_loss=8.9811, test_loss=2.2172
[Epoch 4] train_loss=8.6674, test_loss=2.1516
[Epoch 5] train_loss=8.3056, test_loss=2.0755
[Epoch 6] train_loss=8.0187, test_loss=2.0391
[Epoch 7] train_loss=7.8970, test_loss=2.0228
[Epoch 8] train_loss=7.7060, test_loss=1.9450
[Epoch 9] train_loss=7.5420, test_loss=1.9124
[Epoch 10] train_loss=7.4388, test_loss=1.8706
[Epoch 11] train_loss=7.3237, test_loss=1.8657
[Epoch 12] train_loss=7.2773, test_loss=1.8357
[Epoch 13] train_loss=7.2123, test_loss=1.8522
[Epoch 14] train_loss=7.1902, test_loss=1.8117
[Epoch 15] train_loss=7.0730, test_loss=1.7966
[Epoch 16] train_loss=7.0306, test_loss=1.7909
[Epoch 17] train_loss=7.0123, test_loss=1.7852
[Epoch 18] train_loss=7.0017, test_loss=1.7873
[Epoch 19] train_loss=7.0002, test_loss=1.7882
[Epoch 20] train_loss=6.9848, test_loss=1.7774

=== INN/GLOW ATTACK COMPLETE ===
Final test_loss: 1.777389407157898
If test_loss remains O(1), inversion is NOT learnable.

=== TBJA-SURROGATE START (n=32, steps=1000) ===
  0%|                                                                                                                                        | 0/1000 [00:00<?, ?it/s][TBJA] iter=0  loss=0.747058
 20%|████████████████████████▊                                                                                                    | 198/1000 [00:00<00:02, 267.64it/s][TBJA] iter=200  loss=4.076149
 38%|████████████████████████████████████████████████                                                                             | 384/1000 [00:01<00:02, 253.24it/s][TBJA] iter=400  loss=12.309236
 58%|████████████████████████████████████████████████████████████████████████▊                                                    | 582/1000 [00:02<00:01, 219.20it/s][TBJA] iter=600  loss=20.333547
 79%|███████████████████████████████████████████████████████████████████████████████████████████████████▎                         | 794/1000 [00:03<00:00, 232.46it/s][TBJA] iter=800  loss=25.595000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:04<00:00, 230.63it/s]

=== TBJA-SURROGATE RESULT ===
Final loss = 28.315127433100557



(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_wct_modes_tbja.py

=== WCT MODES-TBJA START (N=32, K=8, T_pde=40, steps=600) ===
  0%|                                                                                                                                         | 0/600 [00:00<?, ?it/s][WCT-TBJA] iter=   0  L_main=0.116489  L_curv=342.081624  L_total=342.198113
  8%|██████████▋                                                                                                                     | 50/600 [02:17<26:59,  2.94s/it][WCT-TBJA] iter=  50  L_main=0.116478  L_curv=342.081606  L_total=342.198084
WCT-TBJA] iter=  50  L_main=0.116478  L_curv=342.081606  L_total=342.198084
 17%|█████████████████████▏                                                                                                         | 100/600 [04:34<09:46,  1.17s/it][WCT-TBJA] iter= 100  L_main=0.115976  L_curv=342.075132  L_total=342.191108
 25%|███████████████████████████████▊                                                                                               | 150/600 [07:09<25:49,  3.44s/it][WCT-TBJA] iter= 150  L_main=0.114334  L_curv=341.030490  L_total=341.144824
 33%|██████████████████████████████████████████▎                                                                                    | 200/600 [09:59<24:02,  3.61s/it][WCT-TBJA] iter= 200  L_main=0.100168  L_curv=201.219132  L_total=201.319300
 42%|████████████████████████████████████████████████████▉                                                                          | 250/600 [12:37<19:31,  3.35s/it][WCT-TBJA] iter= 250  L_main=0.133551  L_curv=0.143578  L_total=0.277129
 50%|███████████████████████████████████████████████████████████████▌                                                               | 300/600 [15:57<16:00,  3.20s/it][WCT-TBJA] iter= 300  L_main=0.134658  L_curv=0.000062  L_total=0.134720
 58%|██████████████████████████████████████████████████████████████████████████                                                     | 350/600 [18:24<06:12,  1.49s/it][WCT-TBJA] iter= 350  L_main=0.134634  L_curv=0.000002  L_total=0.134636
 67%|████████████████████████████████████████████████████████████████████████████████████▋                                          | 400/600 [19:32<08:40,  2.60s/it][WCT-TBJA] iter= 400  L_main=0.134619  L_curv=0.000001  L_total=0.134620
 75%|███████████████████████████████████████████████████████████████████████████████████████████████▎                               | 450/600 [20:33<04:40,  1.87s/it][WCT-TBJA] iter= 450  L_main=0.134610  L_curv=0.000002  L_total=0.134613
 83%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▊                     | 500/600 [21:42<01:26,  1.15it/s][WCT-TBJA] iter= 500  L_main=0.134603  L_curv=0.000002  L_total=0.134605
 92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍          | 550/600 [22:44<01:12,  1.46s/it][WCT-TBJA] iter= 550  L_main=0.134593  L_curv=0.000002  L_total=0.134595
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [23:32<00:00,  2.35s/it]

=== WCT MODES-TBJA RESULT ===
Best total loss : 0.13458797906584624
Final correlation(ψ0_best, ψ0_true) : 0.003643455764732744



(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_sha_collision.py
=== SHA256 Collision Cluster: n=6, seeds=[0,100000) with 4 workers ===
[Worker 0-25000] seed=0, local_size=1
[Worker 50000-75000] seed=50000, local_size=1
[Worker 75000-100000] seed=75000, local_size=1
[Worker 25000-50000] seed=25000, local_size=1
[Worker 50000-75000] seed=51000, local_size=1001
[Worker 0-25000] seed=1000, local_size=1001
[Worker 25000-50000] seed=26000, local_size=1001
[Worker 75000-100000] seed=76000, local_size=1001
[Worker 75000-100000] seed=77000, local_size=2001
[Worker 25000-50000] seed=27000, local_size=2001
[Worker 50000-75000] seed=52000, local_size=2001
[Worker 0-25000] seed=2000, local_size=2001
[Worker 25000-50000] seed=28000, local_size=3001
[Worker 75000-100000] seed=78000, local_size=3001
[Worker 50000-75000] seed=53000, local_size=3001
[Worker 0-25000] seed=3000, local_size=3001
[Worker 75000-100000] seed=79000, local_size=4001
[Worker 25000-50000] seed=29000, local_size=4001
[Worker 0-25000] seed=4000, local_size=4001
[Worker 50000-75000] seed=54000, local_size=4001
[Worker 75000-100000] seed=80000, local_size=5001
[Worker 25000-50000] seed=30000, local_size=5001
[Worker 0-25000] seed=5000, local_size=5001
[Worker 50000-75000] seed=55000, local_size=5001
[Worker 75000-100000] seed=81000, local_size=6001
[Worker 0-25000] seed=6000, local_size=6001
[Worker 25000-50000] seed=31000, local_size=6001
[Worker 50000-75000] seed=56000, local_size=6001
[Worker 75000-100000] seed=82000, local_size=7001
[Worker 0-25000] seed=7000, local_size=7001
[Worker 25000-50000] seed=32000, local_size=7001
[Worker 50000-75000] seed=57000, local_size=7001
[Worker 75000-100000] seed=83000, local_size=8001
[Worker 0-25000] seed=8000, local_size=8001
[Worker 25000-50000] seed=33000, local_size=8001
[Worker 50000-75000] seed=58000, local_size=8001
[Worker 75000-100000] seed=84000, local_size=9001
[Worker 0-25000] seed=9000, local_size=9001
[Worker 25000-50000] seed=34000, local_size=9001
[Worker 50000-75000] seed=59000, local_size=9001
[Worker 75000-100000] seed=85000, local_size=10001
[Worker 0-25000] seed=10000, local_size=10001
[Worker 25000-50000] seed=35000, local_size=10001
[Worker 50000-75000] seed=60000, local_size=10001
[Worker 75000-100000] seed=86000, local_size=11001
[Worker 0-25000] seed=11000, local_size=11001
[Worker 25000-50000] seed=36000, local_size=11001
[Worker 50000-75000] seed=61000, local_size=11001
[Worker 75000-100000] seed=87000, local_size=12001
[Worker 0-25000] seed=12000, local_size=12001
[Worker 25000-50000] seed=37000, local_size=12001
[Worker 50000-75000] seed=62000, local_size=12001
[Worker 75000-100000] seed=88000, local_size=13001
[Worker 0-25000] seed=13000, local_size=13001
[Worker 25000-50000] seed=38000, local_size=13001
[Worker 50000-75000] seed=63000, local_size=13001
[Worker 75000-100000] seed=89000, local_size=14001
[Worker 0-25000] seed=14000, local_size=14001
[Worker 25000-50000] seed=39000, local_size=14001
[Worker 50000-75000] seed=64000, local_size=14001
[Worker 75000-100000] seed=90000, local_size=15001
[Worker 0-25000] seed=15000, local_size=15001
[Worker 25000-50000] seed=40000, local_size=15001
[Worker 50000-75000] seed=65000, local_size=15001
[Worker 0-25000] seed=16000, local_size=16001
[Worker 75000-100000] seed=91000, local_size=16001
[Worker 25000-50000] seed=41000, local_size=16001
[Worker 50000-75000] seed=66000, local_size=16001
[Worker 0-25000] seed=17000, local_size=17001
[Worker 75000-100000] seed=92000, local_size=17001
[Worker 25000-50000] seed=42000, local_size=17001
[Worker 50000-75000] seed=67000, local_size=17001
[Worker 0-25000] seed=18000, local_size=18001
[Worker 75000-100000] seed=93000, local_size=18001
[Worker 25000-50000] seed=43000, local_size=18001
[Worker 50000-75000] seed=68000, local_size=18001
[Worker 25000-50000] seed=44000, local_size=19001
[Worker 75000-100000] seed=94000, local_size=19001
[Worker 0-25000] seed=19000, local_size=19001
[Worker 50000-75000] seed=69000, local_size=19001
[Worker 25000-50000] seed=45000, local_size=20001
[Worker 75000-100000] seed=95000, local_size=20001
[Worker 0-25000] seed=20000, local_size=20001
[Worker 50000-75000] seed=70000, local_size=20001
[Worker 25000-50000] seed=46000, local_size=21001
[Worker 75000-100000] seed=96000, local_size=21001
[Worker 0-25000] seed=21000, local_size=21001
[Worker 50000-75000] seed=71000, local_size=21001
[Worker 25000-50000] seed=47000, local_size=22001
[Worker 0-25000] seed=22000, local_size=22001
[Worker 75000-100000] seed=97000, local_size=22001
[Worker 50000-75000] seed=72000, local_size=22001
[Worker 25000-50000] seed=48000, local_size=23001
[Worker 0-25000] seed=23000, local_size=23001
[Worker 75000-100000] seed=98000, local_size=23001
[Worker 50000-75000] seed=73000, local_size=23001
[Worker 25000-50000] seed=49000, local_size=24001
[Worker 0-25000] seed=24000, local_size=24001
[Worker 75000-100000] seed=99000, local_size=24001
[Worker 50000-75000] seed=74000, local_size=24001

=== Collision Search Result ===
Checked 100000 seeds.
No collisions found in searched range.







(cupy-env) PS C:\Users\Ricky.Reyes.CST\Desktop\Wavelock\tests> python .\test_wavelock_advanced.py

============================================================
  MULTI-DEPTH PDE EVOLUTION TEST (FAST)
============================================================

[WL-TEST] ==== T = 5 ====
[WL-TEST] Generating spectral surrogate seed n=32, seed=970684319
[WL-TEST] ψ0 spectral surrogate, shape=(32, 32)
[WL-TEST] Calling evolve() twice for determinism...
[WL-TEST] evolve() pair completed in 0.359 sec
  Determinism: PASS
  Output norm = 9779.524033
  Hash(T=5)   = 1e03f43722a614a8...
  Hash(T=6) = 1e03f43722a614a8...
  Depth hash variation: FAIL
[WL-TEST] ==== T = 10 ====
[WL-TEST] Generating spectral surrogate seed n=32, seed=663739864
[WL-TEST] ψ0 spectral surrogate, shape=(32, 32)
[WL-TEST] Calling evolve() twice for determinism...
[WL-TEST] evolve() pair completed in 0.113 sec
  Determinism: PASS
  Output norm = 81019.402595
  Hash(T=10)   = 61ce4168ad14f037...
  Hash(T=11) = 61ce4168ad14f037...
  Depth hash variation: FAIL
[WL-TEST] ==== T = 20 ====
[WL-TEST] Generating spectral surrogate seed n=32, seed=556839919
[WL-TEST] ψ0 spectral surrogate, shape=(32, 32)
[WL-TEST] Calling evolve() twice for determinism...
[WL-TEST] evolve() pair completed in 0.179 sec
  Determinism: PASS
  Output norm = 171147.984250
  Hash(T=20)   = 5aea062f14477854...
  Hash(T=21) = 5aea062f14477854...
  Depth hash variation: FAIL

============================================================
  STRUCTURED SEED INVERSION TEST (FAST)
============================================================

[WL-TEST] === Pattern: sin ===
[WL-TEST] evolve(psi0, T=10) starting...
[WL-TEST] evolve() completed in 0.064 sec
  True hash  = d78277bb44f7fd14...
  Guess hash = 2726c3f5dee092fb...
  Structured inversion resistance: PASS
[WL-TEST] === Pattern: bump ===
[WL-TEST] evolve(psi0, T=10) starting...
[WL-TEST] evolve() completed in 0.056 sec
  True hash  = 8230d9a3b6d70bb9...
  Guess hash = f17271e04de471fb...
  Structured inversion resistance: PASS
[WL-TEST] === Pattern: radial ===
[WL-TEST] evolve(psi0, T=10) starting...
[WL-TEST] evolve() completed in 0.060 sec
  True hash  = 53f5dfb6ed8d19eb...
  Guess hash = 8ed12438fbb4c45f...
  Structured inversion resistance: PASS
[WL-TEST] === Pattern: chess ===
[WL-TEST] evolve(psi0, T=10) starting...
[WL-TEST] evolve() completed in 0.059 sec
  True hash  = 94a55694d1a38b33...
  Guess hash = a98a0b5c504e181d...
  Structured inversion resistance: PASS

============================================================
  MULTI-RESOLUTION SPLIT INVERSION (FAST)
============================================================

[WL-TEST] Generating spectral surrogate seed n=32, seed=390663865
[WL-TEST] ψ0 spectral surrogate, shape=(32, 32)
[WL-TEST] evolve(psi0, 10) starting...
[WL-TEST] evolve() completed in 0.059 sec
  True hash = b272559af13e0be0...
  Est hash  = 9f1dcbc35c350d60...
  Multi-res inversion resistance: PASS

============================================================
  RANDOM PROJECTION JACOBIAN TEST (FAST)
============================================================

[WL-TEST] Generating spectral surrogate seed n=8, seed=687678538
[WL-TEST] ψ0 spectral surrogate, shape=(8, 8)
[WL-TEST] Direction 1/6
[WL-TEST] evolve(+/- eps) took 0.109 sec
  contraction ratio = 73025083.597729
  FAIL
[WL-TEST] Direction 2/6
[WL-TEST] evolve(+/- eps) took 0.124 sec
  contraction ratio = 29970163.755480
  FAIL
[WL-TEST] Direction 3/6
[WL-TEST] evolve(+/- eps) took 0.135 sec
  contraction ratio = 32711265.298588
  FAIL
[WL-TEST] Direction 4/6
[WL-TEST] evolve(+/- eps) took 0.123 sec
  contraction ratio = 8423820.967211
  FAIL
[WL-TEST] Direction 5/6
[WL-TEST] evolve(+/- eps) took 0.116 sec
  contraction ratio = 5989946.166764
  FAIL
[WL-TEST] Direction 6/6
[WL-TEST] evolve(+/- eps) took 0.117 sec
  contraction ratio = 180558393.187359
  FAIL

============================================================
  STOCHASTIC ADJOINT INVERSION (FAST)
============================================================

[WL-TEST] Generating spectral surrogate seed n=16, seed=339880631
[WL-TEST] ψ0 spectral surrogate, shape=(16, 16)
[WL-TEST] Running noisy adjoint descent (40 steps)...
[WL-TEST]   iter 00 evolve pair: 0.123 sec
[WL-TEST]   iter 01 evolve pair: 0.123 sec
[WL-TEST]   iter 02 evolve pair: 0.125 sec
[WL-TEST]   iter 03 evolve pair: 0.121 sec
[WL-TEST]   iter 04 evolve pair: 0.121 sec
[WL-TEST]   iter 05 evolve pair: 0.127 sec
[WL-TEST]   iter 06 evolve pair: 0.115 sec
[WL-TEST]   iter 07 evolve pair: 0.118 sec
[WL-TEST]   iter 08 evolve pair: 0.123 sec
[WL-TEST]   iter 09 evolve pair: 0.115 sec
[WL-TEST]   iter 10 evolve pair: 0.116 sec
[WL-TEST]   iter 11 evolve pair: 0.112 sec
[WL-TEST]   iter 12 evolve pair: 0.111 sec
[WL-TEST]   iter 13 evolve pair: 0.124 sec
[WL-TEST]   iter 14 evolve pair: 0.126 sec
[WL-TEST]   iter 15 evolve pair: 0.118 sec
[WL-TEST]   iter 16 evolve pair: 0.119 sec
[WL-TEST]   iter 17 evolve pair: 0.119 sec
[WL-TEST]   iter 18 evolve pair: 0.116 sec
[WL-TEST]   iter 19 evolve pair: 0.113 sec
[WL-TEST]   iter 20 evolve pair: 0.114 sec
[WL-TEST]   iter 21 evolve pair: 0.118 sec
[WL-TEST]   iter 22 evolve pair: 0.113 sec
[WL-TEST]   iter 23 evolve pair: 0.112 sec
[WL-TEST]   iter 24 evolve pair: 0.119 sec
[WL-TEST]   iter 25 evolve pair: 0.116 sec
[WL-TEST]   iter 26 evolve pair: 0.117 sec
[WL-TEST]   iter 27 evolve pair: 0.116 sec
[WL-TEST]   iter 28 evolve pair: 0.114 sec
[WL-TEST]   iter 29 evolve pair: 0.114 sec
[WL-TEST]   iter 30 evolve pair: 0.117 sec
[WL-TEST]   iter 31 evolve pair: 0.120 sec
[WL-TEST]   iter 32 evolve pair: 0.112 sec
[WL-TEST]   iter 33 evolve pair: 0.112 sec
[WL-TEST]   iter 34 evolve pair: 0.117 sec
[WL-TEST]   iter 35 evolve pair: 0.117 sec
[WL-TEST]   iter 36 evolve pair: 0.115 sec
[WL-TEST]   iter 37 evolve pair: 0.114 sec
[WL-TEST]   iter 38 evolve pair: 0.118 sec
[WL-TEST]   iter 39 evolve pair: 0.113 sec
  True hash  = 70e3cabbe389bb78...
  Guess hash = fb6ecad6a91575d8...
  Stochastic inversion: PASS

============================================================
  GLOBAL CONTRACTION TEST (FAST)
============================================================

[WL-TEST] Generating spectral surrogate seed n=16, seed=451031520
[WL-TEST] ψ0 spectral surrogate, shape=(16, 16)
[WL-TEST] Generating spectral surrogate seed n=16, seed=153530018
[WL-TEST] ψ0 spectral surrogate, shape=(16, 16)
[WL-TEST] evolve(a,b, T=10) starting...
[WL-TEST] evolve pair completed in 0.114 sec
  Initial d0 = 7.460395
  Output  dT = 11739.300818
  Global contraction: FAIL

=============== ALL ADVANCED TESTS COMPLETE (FAST MODE) ===============