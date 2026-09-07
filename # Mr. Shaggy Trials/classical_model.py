"""
=============================================================================
CLASSICAL MODEL — Baseline for 'The 64-Copy Problem'
Alexandria Quantum Hackathon 2026 — Team SHAGGY
=============================================================================

PURPOSE:
    This is the CLASSICAL BASELINE required by the competition report.
    It uses ONLY computational-basis measurements — NO quantum circuit
    is applied before measurement. The oracle is called as:
        oracle.measure(state_id)          # raw computational-basis sample
    and NEVER as:
        oracle.measure(state_id, ops_fn=...)   # <-- NOT used here

    This model proves that a purely classical approach (threshold-based
    decision rules on physics order parameters) works reasonably at high
    copy counts but DEGRADES sharply in the starved regime (k ≤ 16),
    which is exactly what the quantum model is designed to beat.

WHY THIS MATTERS FOR THE REPORT:
    The competition rules (Section 4 of RULES.md) state:
        "Classical baselines in your report must obey the same 64-copy
         budget as your main protocol — that is what makes your comparison
         meaningful."
    This file IS that budget-matched classical baseline.

HOW IT WORKS — Step by Step:
    1. For each unseen state, we consume `k` copies in the computational
       basis (where k = oracle.remaining(sid), respecting the budget).
    2. From the raw bitstrings {0,1}^16, we compute 5 physics-based
       "order parameters" — scalar numbers that characterize the phase:

       a) M_z  (Total Magnetization):
          • Formula: M_z = (1/N) × Σ_i Z_i,  where Z_i = 1 - 2×bit_i
          • FM states: M_z ≈ +1.0 (all spins up → all bits = 0)
          • XY & NEEL states: M_z ≈ 0.0 (half the bits are 0, half are 1)
          • WHY: The tiny field h = 1e-4 along +z breaks the degeneracy
            in the FM phase, forcing the ground state to be |000...0⟩.

       b) M_stag² (Staggered Magnetization Squared):
          • Formula: M_stag = (1/N) × Σ_i (-1)^i × Z_i
                     M_stag² = ⟨M_stag²⟩  (averaged over shots)
          • NEEL states: M_stag² ∈ [0.35, 0.73] — spins alternate ↑↓↑↓
          • XY states:   M_stag² ∈ [0.05, 0.20] — weaker, algebraic decay
          • WHY: The Néel phase has long-range antiferromagnetic order.
            Measuring (-1)^i × Z_i picks up this alternating pattern.

       c) C_zz(1) (Nearest-Neighbor Spin-Spin Correlation):
          • Formula: C_zz = (1/N) × Σ_i Z_i × Z_{i+1}  (periodic ring)
          • NEEL states: C_zz ∈ [-0.85, -0.65] (adjacent spins anti-aligned)
          • XY states:   C_zz ∈ [-0.54, -0.23] (moderate anti-correlation)
          • FM states:   C_zz ≈ +1.0 (adjacent spins aligned)
          • WHY: This is the simplest two-point correlator. It measures
            how much neighboring spins agree (+) or disagree (-).

       d) Link Variance (Impostor Detection):
          • Formula: Var_i[⟨Z_i Z_{i+1}⟩]  (variance ACROSS the 16 links)
          • XXZ states: ≈ 0.0 (translation invariance → all links identical)
          • Fermi-Hubbard impostors: >> 0 (Jordan-Wigner boundary at qubit 7-8
            breaks the ring symmetry, creating a "hot link")
          • WHY: The XXZ chain with periodic boundary conditions has exact
            1-site translation symmetry. The Fermi-Hubbard model mapped
            via Jordan-Wigner has qubits 0-7 = spin-up fermions and
            8-15 = spin-down fermions, creating an artificial boundary.

       e) HW Deviation (Hamming Weight Deviation from Half-Filling):
          • Formula: |mean_hamming_weight/N - 0.5|
          • XXZ states (all phases): ≈ 0 (exactly in S_z = 0 sector, 8 ones)
          • Fermi-Hubbard with doping: >> 0 (particle number ≠ N/2)
          • WHY: The XXZ Hamiltonian commutes with total S_z. Ground states
            live in the S_z = 0 sector (exactly 8 spin-ups out of 16).
            Doped Fermi-Hubbard states violate this.

    3. We apply a DECISION TREE of threshold rules:
       - M_z > 0.50  →  "FM"
       - Impostor tests (link_variance > threshold OR hw_dev > threshold)  →  "UNKNOWN"
       - M_stag² and C_zz combined score  →  "NEEL" vs "XY"

BUDGET AWARENESS:
    This model reads oracle.remaining(sid) and uses ALL available copies.
    It works correctly at ANY budget: k = 4, 8, 16, 32, or 64.
    At low k, the order parameter estimates are noisy → accuracy drops.
    This is the fundamental limitation of purely classical measurement:
    you're stuck in the computational basis and can't rotate into better
    measurement bases.

EXPECTED PERFORMANCE (approximate, seed-dependent):
    Budget k=4:   Macro-F1 ≈ 0.45 – 0.65  (very noisy estimates)
    Budget k=8:   Macro-F1 ≈ 0.60 – 0.75
    Budget k=16:  Macro-F1 ≈ 0.75 – 0.88
    Budget k=32:  Macro-F1 ≈ 0.85 – 0.95
    Budget k=64:  Macro-F1 ≈ 0.90 – 0.98

USAGE:
    # Self-score with default 64-copy budget:
    python evaluate.py classical_model.py --data xxz_public_train.npz --key train_answer_key.csv

    # Test at starved budgets:
    python evaluate.py classical_model.py --data xxz_public_train.npz --key train_answer_key.csv --budget 4
    python evaluate.py classical_model.py --data xxz_public_train.npz --key train_answer_key.csv --budget 8
    python evaluate.py classical_model.py --data xxz_public_train.npz --key train_answer_key.csv --budget 16
"""

import numpy as np


# =============================================================================
# FEATURE EXTRACTION — Pure classical, computational-basis only
# =============================================================================
def extract_classical_features(samples):
    """
    Extract 5 physics-based order parameters from raw bitstring measurements.

    Parameters
    ----------
    samples : np.ndarray of shape (num_shots, num_qubits)
        Binary measurement outcomes. Convention: 0 = spin up (+1), 1 = spin down (-1).

    Returns
    -------
    dict with keys:
        'mz'              : float — Total magnetization ⟨M_z⟩
        'm_stag_sq'       : float — Staggered magnetization squared ⟨M_stag²⟩
        'nn_zz'           : float — Nearest-neighbor ZZ correlation ⟨C_zz(1)⟩
        'link_variance'   : float — Variance of per-link correlations (impostor test)
        'hw_fraction_dev' : float — |mean_hamming_weight/N - 0.5| (impostor test)

    Physics Explanation
    -------------------
    We map computational-basis outcomes {0, 1} to Pauli-Z eigenvalues {+1, -1}:
        spin_i = 1 - 2 × bit_i

    Then all order parameters are standard condensed-matter observables computed
    from these spin values. No quantum circuit is needed — this is a purely
    classical statistical analysis of raw measurement data.
    """
    samples = np.asarray(samples, dtype=np.float64)
    num_shots, num_qubits = samples.shape

    # ---- Map bits → spins: {0,1} → {+1,-1} ----
    spins = 1.0 - 2.0 * samples

    # ---- 1. Total Magnetization M_z ----
    # Average spin value across ALL qubits and ALL shots.
    # FM: ≈ +1.0 | XY: ≈ 0.0 | NEEL: ≈ 0.0
    mz = float(np.mean(spins))

    # ---- 2. Staggered Magnetization Squared M_stag² ----
    # Apply (-1)^i sign to each site, then compute per-shot staggered magnetization.
    # Square it and average over shots to get ⟨M_stag²⟩.
    # NEEL: [0.35, 0.73] | XY: [0.05, 0.20] | FM: ≈ 0
    staggered_sign = np.array([(-1.0) ** i for i in range(num_qubits)])
    m_stag_per_shot = np.sum(spins * staggered_sign, axis=1) / float(num_qubits)
    m_stag_sq = float(np.mean(m_stag_per_shot ** 2))

    # ---- 3. Nearest-Neighbor Correlation C_zz(1) ----
    # For each link (i, i+1 mod N), compute ⟨Z_i × Z_{i+1}⟩ averaged over shots.
    # Then average over all N links.
    # NEEL: [-0.85, -0.65] | XY: [-0.54, -0.23] | FM: ≈ +1.0
    spins_shifted = np.roll(spins, -1, axis=1)  # site i+1 (periodic)
    per_link_corr = np.mean(spins * spins_shifted, axis=0)  # shape (N,)
    nn_zz = float(np.mean(per_link_corr))

    # ---- 4. Link Variance (Impostor Detection) ----
    # Variance of per-link correlations ACROSS the ring.
    # XXZ (translation invariant): ≈ 0 (all links identical by symmetry)
    # Fermi-Hubbard (JW mapped): >> 0 (link 7↔8 is an artificial boundary)
    link_variance = float(np.var(per_link_corr))

    # ---- 5. Hamming Weight Deviation ----
    # XXZ ground states have exactly N/2 spin-ups (S_z = 0 sector).
    # Mean hamming weight / N should be 0.5. Deviation signals doping.
    mean_hw_fraction = float(np.mean(np.sum(samples, axis=1))) / float(num_qubits)
    hw_fraction_dev = abs(mean_hw_fraction - 0.5)

    return {
        "mz": mz,
        "m_stag_sq": m_stag_sq,
        "nn_zz": nn_zz,
        "link_variance": link_variance,
        "hw_fraction_dev": hw_fraction_dev,
    }


# =============================================================================
# CLASSIFIER — Pure threshold-based decision tree (no ML training needed)
# =============================================================================
def classify(oracle, state_ids):
    """
    Classical baseline classifier using ONLY computational-basis measurements.

    This function obeys the competition contract:
        classify(oracle, state_ids) → {state_id: label}
        label ∈ {"FM", "XY", "NEEL", "UNKNOWN"}

    Decision Logic (hand-crafted thresholds based on physics):
    ──────────────────────────────────────────────────────────
    Step 1: Measure ALL available copies in computational basis.
    Step 2: Compute 5 order parameters.
    Step 3: Apply decision rules:

        ┌─ M_z > 0.50 ───────────────────────────────► "FM"
        │
        ├─ link_variance > threshold ─────────────────► "UNKNOWN"
        ├─ hw_fraction_dev > threshold AND |M_z| < 0.40 ► "UNKNOWN"
        │
        └─ combined_score > 0.55 ─────────────────────► "NEEL"
           else ──────────────────────────────────────► "XY"

    The thresholds adapt based on the copy budget:
    - At low k (4-8): We use RELAXED thresholds because estimates are noisy.
    - At high k (32-64): We use TIGHTER thresholds for better precision.

    Parameters
    ----------
    oracle : CopyOracle
        The metered oracle that provides quantum state measurements.
    state_ids : list of str
        State identifiers to classify.

    Returns
    -------
    dict : {state_id: label}
    """
    predictions = {}

    for sid in state_ids:
        # ── How many copies do we have? ──
        budget = oracle.remaining(sid)

        # ── Consume ALL available copies in computational basis ──
        # No quantum circuit — just raw Z-basis measurements.
        samples = []
        for _ in range(budget):
            samples.append(oracle.measure(sid))
        samples = np.array(samples)

        # ── Extract classical features ──
        features = extract_classical_features(samples)

        # ── Adapt thresholds to budget ──
        # At low shot counts, statistical fluctuations are large.
        # We relax thresholds to avoid false classifications.
        if budget <= 4:
            fm_threshold = 0.55       # Relaxed: few shots → noisy M_z
            impostor_link_var = 0.60   # Relaxed: link variance noisy at k=4
            impostor_hw_dev = 0.18     # Relaxed: hamming weight noisy
            neel_threshold = 0.50      # Relaxed: staggered mag noisy
        elif budget <= 8:
            fm_threshold = 0.55
            impostor_link_var = 0.55
            impostor_hw_dev = 0.16
            neel_threshold = 0.52
        elif budget <= 16:
            fm_threshold = 0.50
            impostor_link_var = 0.50
            impostor_hw_dev = 0.15
            neel_threshold = 0.55
        else:
            fm_threshold = 0.50        # Tight: many shots → stable estimates
            impostor_link_var = 0.48
            impostor_hw_dev = 0.14
            neel_threshold = 0.55

        # ── STEP 1: Ferromagnetic Detection ──
        # FM is the easiest phase to detect classically.
        # All spins point along +z → M_z ≈ +1.0 even with noise.
        if features["mz"] > fm_threshold:
            predictions[sid] = "FM"
            continue

        # ── STEP 2: Impostor Detection ──
        # Two independent tests for Fermi-Hubbard impostor states:
        is_impostor = False

        # Test A: Link variance — does the ring have a "hot link"?
        if features["link_variance"] > impostor_link_var:
            is_impostor = True

        # Test B: Hamming weight deviation — is it away from half-filling?
        # Only apply when M_z is small (to avoid false positive on FM)
        if features["hw_fraction_dev"] > impostor_hw_dev and abs(features["mz"]) < 0.40:
            is_impostor = True

        if is_impostor:
            predictions[sid] = "UNKNOWN"
            continue

        # ── STEP 3: NEEL vs XY Discrimination ──
        # This is the HARDEST part for the classical model.
        # Both phases have M_z ≈ 0, but NEEL has stronger staggered order
        # and more negative nearest-neighbor correlations.
        #
        # combined_score merges both signals:
        #   M_stag² is LARGE for NEEL, SMALL for XY
        #   C_zz(1) is VERY NEGATIVE for NEEL, MODERATELY NEGATIVE for XY
        #   → subtracting nn_zz (which is negative) ADDS to the score for NEEL
        combined_score = features["m_stag_sq"] - 0.50 * features["nn_zz"]

        if combined_score > neel_threshold:
            predictions[sid] = "NEEL"
        else:
            predictions[sid] = "XY"

    return predictions


# =============================================================================
# STANDALONE TEST — Run this file directly to see it in action
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from oracle import CopyOracle

    print("=" * 70)
    print("CLASSICAL MODEL — Standalone Test")
    print("=" * 70)

    data = np.load("xxz_public_train.npz", allow_pickle=True)
    true_labels = {str(sid): str(lab) for sid, lab in zip(data["ids"], data["labels"])}

    # Test at multiple budgets to show degradation
    for budget in [4, 8, 16, 32, 64]:
        oracle = CopyOracle(
            "xxz_public_train.npz", copy_budget=budget, noise_p=0.02, seed=42
        )
        preds = classify(oracle, oracle.state_ids())

        correct = sum(1 for sid in preds if preds[sid] == true_labels.get(sid, ""))
        total = len(preds)
        print(f"  Budget k={budget:2d}:  {correct}/{total} correct  "
              f"(accuracy = {correct/total:.2f})")

    print("\nTo get full scoring with macro-F1, run:")
    print("  python evaluate.py classical_model.py --data xxz_public_train.npz "
          "--key train_answer_key.csv --budget 64")
