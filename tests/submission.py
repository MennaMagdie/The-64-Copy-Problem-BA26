"""
submission.py - Final Variational QNN Submission
Architecture: Dedicated Z-Basis Projective Measurement -> 4 QFE Order Parameters -> 4-Qubit Variational QNN
Interface: def classify(oracle, state_ids) -> {state_id: label}
"""
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# =====================================================================
# 1. 4-QUBIT VARIATIONAL QUANTUM NEURAL NETWORK
# =====================================================================
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnn_circuit(inputs, weights):
    # Angle Embedding: Rotate 4 qubits around Y axis by scaled features
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

    # Strongly Entangling Layers
    for l in range(weights.shape[0]):
        for q in range(n_qubits):
            qml.RY(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class HybridQNN(nn.Module):
    def __init__(self, n_layers=2, n_classes=4):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers, n_qubits, 2))
        self.fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        batch_out = []
        for sample in x:
            q_out = qnn_circuit(sample, self.weights)
            batch_out.append(torch.stack(q_out))
        q_features = torch.stack(batch_out).float()
        return self.fc(q_features)


# =====================================================================
# 2. EMBEDDED TRAINED WEIGHTS & SCALING CONSTANTS
# =====================================================================
CLASSES = ["FM", "NEEL", "UNKNOWN", "XY"]

SCALER_MIN = np.array([0.0, 0.0, -0.9375, 0.0], dtype=np.float32)
SCALER_MAX = np.array([1.0, 0.9375, 1.0, 2.0], dtype=np.float32)

WEIGHTS_Q = [
    [
        [0.19378338754177094, -0.24483801424503326],
        [-0.1428600698709488, -0.3902990221977234],
        [0.4464204013347626, -0.3929564356803894],
        [0.12493421137332916, -0.22484007477760315]
    ],
    [
        [0.005350284278392792, 0.15957428514957428],
        [-0.2509661912918091, -0.1416364461183548],
        [0.31773996353149414, -0.0540611557662487],
        [0.0016869530081748962, 0.0769653469324112]
    ]
]

FC_W = [
    [-0.012836471199989319, 0.47419941425323486, 0.15907107293605804, 0.5244291424751282],
    [-0.45496729016304016, 0.0877743735909462, -0.5885122418403625, 0.4769926369190216],
    [-0.16695834696292877, -0.6106457710266113, -0.5414416790008545, -0.6994732618331909],
    [0.4469775855541229, -0.35773906111717224, 0.8352292776107788, -0.18982574343681335]
]

FC_B = [-0.0077401623129844666, -0.08151544630527496, 0.23827394843101501, 0.6304274797439575]

# Instantiate model and load parameters into memory
model = HybridQNN(n_layers=2, n_classes=4)
with torch.no_grad():
    model.weights.copy_(torch.tensor(WEIGHTS_Q, dtype=torch.float32))
    model.fc.weight.copy_(torch.tensor(FC_W, dtype=torch.float32))
    model.fc.bias.copy_(torch.tensor(FC_B, dtype=torch.float32))
model.eval()


# =====================================================================
# 3. MEASUREMENT EXTRACTION & INFERENCE
# =====================================================================
def extract_features(bits):
    """
    Computes 4 physical order parameters from bitstrings.
    Size-invariant: dynamically evaluates over spins.shape[1].
    """
    spins = 1.0 - 2.0 * bits.astype(float)
    n_sites = spins.shape[1]

    # 1. Uniform Magnetization |M_z|
    mz = np.mean(np.abs(np.mean(spins, axis=1)))

    # 2. Staggered Magnetization M_stagg
    stagg_weights = (-1.0) ** np.arange(n_sites)
    stagg = np.mean(np.abs(np.mean(spins * stagg_weights, axis=1)))

    # 3. Nearest-Neighbor Exchange Correlator C_zz
    czz = spins * np.roll(spins, -1, axis=1)
    mean_czz = np.mean(czz)

    # 4. Dimerization (Bond Alternation)
    even_bonds = np.mean(czz[:, 0::2])
    odd_bonds = np.mean(czz[:, 1::2])
    dimer = np.abs(even_bonds - odd_bonds)

    return np.array([mz, stagg, mean_czz, dimer], dtype=np.float32)


def scale_features(feat):
    """Scales extracted order parameters to [0, pi] using training statistics."""
    denom = SCALER_MAX - SCALER_MIN
    denom = np.where(denom == 0.0, 1.0, denom)
    scaled = (feat - SCALER_MIN) / denom * np.pi
    return np.clip(scaled, 0.0, np.pi)


def classify(oracle, state_ids):
    """
    Competition entry point.
    Returns {state_id: label} for every id in state_ids.
    Consumes 32 copies per state (strictly under the 64-copy limit).
    """
    results = {}
    SHOT_BUDGET = 32

    for sid in state_ids:
        # Query oracle for projective measurements
        if hasattr(oracle, "measure"):
            try:
                bits = oracle.measure(sid, shots=SHOT_BUDGET)
            except TypeError:
                bits = oracle.measure(sid, count=SHOT_BUDGET)
        elif hasattr(oracle, "sample"):
            bits = oracle.sample(sid, shots=SHOT_BUDGET)
        else:
            bits = oracle(sid, shots=SHOT_BUDGET)

        bits = np.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]

        # 1. Physics feature extraction
        raw_feat = extract_features(bits)

        # 2. Normalization
        scaled_feat = scale_features(raw_feat)
        x_t = torch.tensor(scaled_feat, dtype=torch.float32).unsqueeze(0)

        # 3. Variational QNN Forward Pass
        with torch.no_grad():
            logits = model(x_t)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        results[sid] = CLASSES[pred_idx]

    return results
