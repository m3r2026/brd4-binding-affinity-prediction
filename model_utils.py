"""
BRD4 CPIパイプライン共通モジュール

chembl1.ipynb（学習）とapp.py（推論）の両方から読み込む。
特徴量抽出のロジックが学習時と推論時でズレると予測がおかしくなるため、
必ずこのファイル経由で smiles_to_graph / smiles_to_fingerprint / CPIRegressor を使うこと。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np


# ===== 化合物: SMILES → グラフ構造 =====

def smiles_to_graph(smiles: str):
    """SMILES文字列をtorch_geometric.data.Dataオブジェクトに変換する。
    Noneが返る場合はRDKitでパースできなかった不正なSMILES。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # (1) ノード特徴量（原子ごと、6次元）
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.GetHybridization()),
        ])
    x = torch.tensor(node_features, dtype=torch.float32)

    # (2) エッジ（化学結合、結合種別込み）
    edges = []
    edge_types = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edges += [[i, j], [j, i]]
        edge_types += [[bond_type], [bond_type]]

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.float32)

    # (3) 分子全体の記述子（標準化前の生値、7次元）
    global_feats = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
    ]
    global_x = torch.tensor(global_feats, dtype=torch.float32).unsqueeze(0)  # [1, 7]

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.global_feat = global_x
    return data


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """RFベースライン用: SMILES → Morganフィンガープリント(numpy配列)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def get_descriptor_dict(smiles: str) -> dict:
    """アプリの「特徴量の内訳」表示用に、記述子を人間が読める形の辞書で返す"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "分子量 (MolWt)": round(Descriptors.MolWt(mol), 2),
        "脂溶性 (LogP)": round(Descriptors.MolLogP(mol), 2),
        "水素結合ドナー数": Descriptors.NumHDonors(mol),
        "水素結合アクセプター数": Descriptors.NumHAcceptors(mol),
        "極性表面積 (TPSA)": round(Descriptors.TPSA(mol), 2),
        "回転可能結合数": Descriptors.NumRotatableBonds(mol),
        "芳香環の数": Descriptors.NumAromaticRings(mol),
    }


# ===== モデル定義（学習時と完全に同一構造） =====

class CPIRegressor(nn.Module):

    def __init__(self, node_feature_dim=6, protein_feature_dim=125, global_feature_dim=7,
                 hidden_dim=128, edge_feature_dim=1):
        super(CPIRegressor, self).__init__()

        self.protein_lin = nn.Linear(protein_feature_dim, hidden_dim)

        nn1 = nn.Sequential(nn.Linear(node_feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1, edge_dim=edge_feature_dim)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(nn2, edge_dim=edge_feature_dim)

        self.global_lin = nn.Linear(global_feature_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_type, batch, protein_feat, global_feat):
        p = self.protein_lin(protein_feat)
        p = F.relu(p)

        c = self.conv1(x, edge_index, edge_type)
        c = F.relu(c)
        c = self.conv2(c, edge_index, edge_type)
        c = F.relu(c)
        c = global_mean_pool(c, batch)

        g = self.global_lin(global_feat)
        g = F.relu(g)

        combined = torch.cat([c, p, g], dim=1)
        out = self.fc1(combined)
        out = F.relu(out)
        out = self.fc2(out)

        return out.squeeze()
