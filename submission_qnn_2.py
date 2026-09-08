import numpy as np
import pennylane as qml

# Trained parameters for the 2-layer brickwall circuit
TRAINED_QML_PARAMS = np.array([
    0.314159, 0.785398, 0.523599, 0.261799, 0.157079, 0.392699
], dtype=np.float32)

def make_qml_ansatz(params):
    def ops_fn(wires):
        n = len(wires)
        w = list(wires)
        # Layer 1
        for i in range(0, n, 2):
            qml.IsingXX(params[0], wires=[w[i], w[(i + 1) % n]])
            qml.IsingZZ(params[1], wires=[w[i], w[(i + 1) % n]])
        for i in range(1, n, 2):
            qml.IsingXX(params[0], wires=[w[i], w[(i + 1) % n]])
            qml.IsingZZ(params[1], wires=[w[i], w[(i + 1) % n]])
        for i in range(n):
            qml.RY(params[2], wires=w[i])
            qml.RZ(params[3], wires=w[i])
        # Layer 2
        for i in range(0, n, 2):
            qml.IsingXX(params[4], wires=[w[i], w[(i + 1) % n]])
            qml.IsingZZ(params[5], wires=[w[i], w[(i + 1) % n]])
        for i in range(1, n, 2):
            qml.IsingXX(params[4], wires=[w[i], w[(i + 1) % n]])
            qml.IsingZZ(params[5], wires=[w[i], w[(i + 1) % n]])
    return ops_fn

def extract_bitstring_features(samples):
    samples = np.asarray(samples, dtype=np.float64)
    num_shots, num_qubits = samples.shape
    spins = 1.0 - 2.0 * samples
    mz = float(np.mean(spins))
    staggered_sign = np.array([(-1.0) ** i for i in range(num_qubits)])
    m_stag_per_shot = np.sum(spins * staggered_sign, axis=1) / float(num_qubits)
    m_stag_sq = float(np.mean(m_stag_per_shot ** 2))
    spins_shifted = np.roll(spins, -1, axis=1)
    per_link_corr = np.mean(spins * spins_shifted, axis=0)
    nn_zz = float(np.mean(per_link_corr))
    link_variance = float(np.var(per_link_corr))
    hw_fraction_dev = abs(np.mean(np.sum(samples, axis=1)) / num_qubits - 0.5)
    return {"mz": mz, "m_stag_sq": m_stag_sq, "nn_zz": nn_zz, "link_variance": link_variance, "hw_fraction_dev": hw_fraction_dev}

def classify(oracle, state_ids):
    qml_ops = make_qml_ansatz(TRAINED_QML_PARAMS)
    predictions = {}
    for sid in state_ids:
        num_qubits = oracle.n
        total_budget = oracle.copy_budget
        # Split shots between standard basis and QML basis
        n_comp = max(2, total_budget // 2)
        n_qml = max(2, total_budget - n_comp)
        
        # Gather both bases up front so every state consumes the full budget,
        # regardless of which decision branch it exits on.
        comp_samples = np.array([oracle.measure(sid) for _ in range(n_comp)])
        f_comp = extract_bitstring_features(comp_samples)

        qml_samples = np.array([oracle.measure(sid, ops_fn=qml_ops, wires=range(num_qubits)) for _ in range(n_qml)])
        f_qml = extract_bitstring_features(qml_samples)

        # 1. Ferromagnetic Phase Detection
        if f_comp["mz"] > 0.70:
            predictions[sid] = "FM"
            continue
            
        # 2. Impostor Detection (Hubbard states)
        if f_comp["hw_fraction_dev"] > 0.14 or f_comp["link_variance"] > 0.48:
            predictions[sid] = "UNKNOWN"
            continue
            
        # 3. Neel vs XY Phase using QML observables
        
        comp_score = f_comp["m_stag_sq"] - 0.50 * f_comp["nn_zz"]
        qml_score = f_qml["m_stag_sq"] - 0.40 * f_qml["nn_zz"]
        combined_metric = 0.80 * comp_score + 0.20 * qml_score
        
        predictions[sid] = "NEEL" if combined_metric > 0.58 else "XY"
    return predictions