"""
submission.py - Final Variational QNN Submission with Physics-Gated OOD Detection
Architecture: Dedicated Z-Basis Projective Measurement -> 4 QFE Order Parameters -> Dimer Gate -> 4-Qubit Variational QNN
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
# 2. EMBEDDED RETRAINED WEIGHTS & NORMALIZATION CONSTANTS
# =====================================================================
CLASSES = ["FM", "NEEL", "UNKNOWN", "XY"]

SCALER_MIN = np.array([0.0078125, 0.00390625, -0.8046875, 0.0], dtype=np.float32)
SCALER_MAX = np.array([0.9921875, 0.82421875,  0.984375,  1.578125], dtype=np.float32)

WEIGHTS_Q = [
    [
        [0.021647771820425987, -0.9933652877807617],
        [-0.42405965924263, 0.008257731795310974],
        [0.7186479568481445, 0.09823355823755264],
        [0.08183611929416656, 0.0024353526532649994]
    ],
    [
        [-0.009357539936900139, 0.13068078458309174],
        [-0.10589028149843216, -0.1174241155385971],
        [0.27268925309181213, -0.06284137070178986],
        [0.0028896722942590714, 0.07250688970088959]
    ]
]

FC_W = [
    [-1.5445477962493896, 1.8978369235992432, 1.6880711317062378, 1.5055437088012695],
    [-0.11449836194515228, 1.5580368041992188, -2.0469465255737305, 1.962416172027588],
    [-0.704224705696106, -2.1111443042755127, -1.6266661882400513, -2.6695470809936523],
    [1.792245864868164, -0.3757944405078888, 1.5856059789657593, -0.13247482478618622]
]

FC_B = [-1.3691266775131226, 0.2442331612110138, 1.908928394317627, 0.16633622348308563]

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
    Consumes exactly 32 copies per state (strictly under the 64-copy limit).
    """
    results = {}
    SHOT_BUDGET = 32
    DIMER_THRESHOLD = 0.24  # Calibrated 3-sigma noise cutoff

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
        dimer = raw_feat[3]

        # 2. Physics OOD Gate
        if dimer > DIMER_THRESHOLD:
            results[sid] = "UNKNOWN"
            continue

        # 3. Variational QNN Forward Pass
        scaled_feat = scale_features(raw_feat)
        x_t = torch.tensor(scaled_feat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(x_t)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        results[sid] = CLASSES[pred_idx]

    return results