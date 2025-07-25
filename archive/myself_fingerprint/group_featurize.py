import numpy as np
from rdkit import Chem
import pandas as pd



# --------------------------------------------------
# 1. 準備 197 種 block-level SMARTS
# --------------------------------------------------
from smarts_list import raw_smarts

# --------------------------------------------------
# 2. 過濾並編譯成 RDKit Mol 物件
# --------------------------------------------------
block_keys = []    # 儲存合法的 SMARTS 字串
patterns  = []     # 儲存對應的 RDKit Mol 物件

for smarts in raw_smarts:
    patt = Chem.MolFromSmarts(smarts)
    if patt:
        block_keys.append(smarts)
        patterns.append(patt)
    else:
        print(f"Warning: 無法解析 SMARTS：{smarts}")

assert len(block_keys) == 197, f"目前有效 SMARTS 數：{len(block_keys)}，應為 197"

# --------------------------------------------------
# 3. 定義抽取函式
# --------------------------------------------------
def extract_block_fragments(mol):
    """
    輸入：RDKit Mol 物件 (單體)
    輸出：長度為 len(block_keys) 的 list，
          每項 = 該 block 出現次數 / 原子數
    """
    N = mol.GetNumAtoms()
    feats = []
    for patt in patterns:
        # uniquify=True 可避免重複計到相同子結構
        matches = mol.GetSubstructMatches(patt, uniquify=True)
        feats.append(len(matches) / N)
    return feats

# --------------------------------------------------
# 4. 範例：對整個 DataFrame 計算 block-level 特徵矩陣
# --------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv('train.csv')  # 假設已有 SMILES 欄位
    block_matrix = []
    for smi in df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # 無效 SMILES → 全 0
            block_matrix.append([0.0]*len(block_keys))
        else:
            block_matrix.append(extract_block_fragments(mol))
    X_block = np.array(block_matrix)  # shape = (n_samples, 197)
    print("Block-level 特徵矩陣形狀：", X_block.shape)
    # 如果要和原子級、鏈級再合併，請依序做 np.hstack
