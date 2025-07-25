import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

# 載入三層特徵抽取函式與對應的 keys
from atom_featurize import extract_atomic_fragments, atom_keys
from group_featurize import extract_block_fragments, block_keys
from chain_featurize import extract_chain_features, chain_keys

# 讀入資料，確保有 SMILES 與 frequency 欄位
df = pd.read_csv('train.csv')

# 1. 針對每支分子，各自抽取 atomic, block, chain 特徵
X_atom, X_block, X_chain = [], [], []
for smi in df['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # 無效 SMILES → 全補 0
        X_atom.append([0] * len(atom_keys))
        X_block.append([0] * len(block_keys))
        X_chain.append([0] * len(chain_keys))
    else:
        X_atom.append(extract_atomic_fragments(mol))
        X_block.append(extract_block_fragments(mol))
        X_chain.append(extract_chain_features(mol))

X_atom  = np.array(X_atom)   # shape = (n_samples, 155)
X_block = np.array(X_block)  # shape = (n_samples, 197)
X_chain = np.array(X_chain)  # shape = (n_samples,  59)

# 2. 計算頻率特徵：log10(frequency)
#    假設 df['frequency'] 單位為 Hz，且皆 > 0
logf = np.log10(df['frequency'].values).reshape(-1, 1)  # shape = (n_samples, 1)

# 3. 水平堆疊所有特徵 → 最終 412 維 (155+197+59+1)
X_all = np.hstack([X_atom, X_block, X_chain, logf])

# 4. (選) Min–Max 縮放到 [0,1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)

# 5. 轉成 DataFrame，並加上欄位名稱
columns = atom_keys + block_keys + chain_keys + ['log_frequency']
df_features = pd.DataFrame(X_scaled, columns=columns)

# 6. 儲存或後續使用
df_features.to_csv('features_all_scaled.csv', index=False)
print("最終特徵矩陣 shape:", X_scaled.shape)
