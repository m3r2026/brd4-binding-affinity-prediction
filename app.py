"""
BRD4 結合親和性予測アプリ

SMILES文字列を入力すると、BRD4に対するpIC50をGNNとRandom Forest
ベースラインの両方で予測し、比較・可視化する。
"""

import streamlit as st
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader

from model_utils import (
    smiles_to_graph,
    smiles_to_fingerprint,
    get_descriptor_dict,
    CPIRegressor,
)

ARTIFACT_DIR = "app_artifacts"


# ===== アーティファクトの読み込み(キャッシュ) =====

@st.cache_resource
def load_artifacts():
    gnn_model = CPIRegressor()
    gnn_model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/best_model.pt", map_location="cpu"))
    gnn_model.eval()

    rf_model = joblib.load(f"{ARTIFACT_DIR}/rf_model.pkl")

    scaler = torch.load(f"{ARTIFACT_DIR}/global_scaler.pt", map_location="cpu")
    global_mean, global_std = scaler["mean"], scaler["std"]

    protein_vector = torch.load(f"{ARTIFACT_DIR}/protein_vector.pt", map_location="cpu")

    train_dist = np.load(f"{ARTIFACT_DIR}/train_pic50_distribution.npy")

    return gnn_model, rf_model, global_mean, global_std, protein_vector, train_dist


def predict_gnn(smiles, gnn_model, global_mean, global_std, protein_vector):
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    graph.global_feat = (graph.global_feat - global_mean) / global_std
    graph.protein_feat = protein_vector.unsqueeze(0)

    loader = DataLoader([graph], batch_size=1)
    batch = next(iter(loader))

    with torch.no_grad():
        pred = gnn_model(
            batch.x, batch.edge_index, batch.edge_type,
            batch.batch, batch.protein_feat, batch.global_feat,
        )
    return pred.item()


def predict_rf(smiles, rf_model):
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None
    return rf_model.predict(fp.reshape(1, -1))[0]


# ===== ページ設定 =====

st.set_page_config(page_title="BRD4 結合親和性予測", page_icon="🧬", layout="centered")

st.title("🧬 BRD4 結合親和性予測")
st.markdown(
    "化合物のSMILES文字列を入力すると、BRD4(ブロモドメイン含有タンパク質4)に対する"
    "結合親和性(pIC50)を、GNN(グラフニューラルネットワーク)とRandom Forestの"
    "2つのモデルで予測します。"
)

with st.expander("このアプリについて"):
    st.markdown(
        "- GNN: 化合物のグラフ構造(原子・結合)とタンパク質のSASA特徴量を統合したモデル\n"
        "- Random Forest: Morganフィンガープリントを用いた古典的な機械学習ベースライン\n"
        "- 学習データ: ChEMBLのBRD4活性データ(6,677件、アッセイのカットオフ値を除外済み)\n"
        "- 詳細な設計・評価プロセスはリポジトリのREADMEを参照してください"
    )

try:
    gnn_model, rf_model, global_mean, global_std, protein_vector, train_dist = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    st.error(
        f"アーティファクトの読み込みに失敗しました: {e}\n\n"
        f"`{ARTIFACT_DIR}/` フォルダに以下が揃っているか確認してください: "
        "best_model.pt, rf_model.pkl, global_scaler.pt, protein_vector.pt, "
        "train_pic50_distribution.npy"
    )

# 入力例(JQ1: BRD4の代表的な阻害剤)
EXAMPLE_SMILES = "CC1=C(C)C2=NN=C(N2N=C1C)C1=NC2=C(S1)C(C)=C(C)C(=N2)C(C)(C)C(=O)OC(C)(C)C"

smiles_input = st.text_input(
    "化合物のSMILESを入力",
    value="",
    placeholder=f"例: {EXAMPLE_SMILES}",
)
use_example = st.button("例(JQ1類似構造)を使う")

if use_example:
    smiles_input = EXAMPLE_SMILES

if smiles_input and artifacts_ok:
    mol = Chem.MolFromSmiles(smiles_input)

    if mol is None:
        st.error("SMILESとして解釈できませんでした。文字列を確認してください。")
    else:
        col_img, col_desc = st.columns([1, 1])

        with col_img:
            st.subheader("分子構造")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img)

        with col_desc:
            st.subheader("分子記述子")
            desc = get_descriptor_dict(smiles_input)
            st.table(desc)

        st.subheader("予測結果")
        gnn_pred = predict_gnn(smiles_input, gnn_model, global_mean, global_std, protein_vector)
        rf_pred = predict_rf(smiles_input, rf_model)

        col1, col2 = st.columns(2)
        col1.metric("GNN予測 pIC50", f"{gnn_pred:.2f}")
        col2.metric("Random Forest予測 pIC50", f"{rf_pred:.2f}")

        st.caption(
            "2つのモデルの予測が大きく異なる場合、その化合物が訓練データの分布から"
            "外れている可能性があります。"
        )

        st.subheader("学習データの分布における位置")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(train_dist, bins=30, color="#4C72B0", edgecolor="black", alpha=0.7)
        ax.axvline(gnn_pred, color="red", linestyle="--", label=f"GNN予測: {gnn_pred:.2f}")
        ax.axvline(rf_pred, color="orange", linestyle="--", label=f"RF予測: {rf_pred:.2f}")
        ax.set_xlabel("pIC50")
        ax.set_ylabel("頻度")
        ax.legend()
        st.pyplot(fig)

elif not artifacts_ok:
    st.info("アーティファクトファイルを配置すると、ここで予測が実行できます。")
