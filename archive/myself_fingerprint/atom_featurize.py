import pandas as pd
from rdkit import Chem
import collections

df = pd.read_csv('train.csv')

# coordination numbers 常見範圍：1~6
all_triplets = [f"{i}-{j}-{k}"
                for i in range(1,7)
                for j in range(1,7)
                for k in range(1,7)]
len(all_triplets)  # 共 216 種理論組合


# 1. 建立一個 Counter 物件，用來統計各種 A–B–C 片段出現的頻次
counts = collections.Counter()

# 2. 針對資料框中每一個 SMILES 字串，進行處理
for smi in df['SMILES']:
    # 2.1 用 RDKit 解析 SMILES 成分子物件
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue   # 或跳過這筆、或丟全部 0 向量
    
    # 2.2 計算每個原子的「配位數」（coordination number）＝它相連的原子數
    #     coord 是個字典：{ atom_index: 配位數, ... }
    coord = {a.GetIdx(): len(a.GetNeighbors()) for a in mol.GetAtoms()}
    
    # 3. 遍歷分子裡的每一條化學鍵 bond，組成 A–B–C 三原子結構
    for bnd in mol.GetBonds():
        # 3.1 取出鍵的兩端原子索引 a, b
        a, b = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()
        
        # 3.2 為了不漏掉 A–B–C 和 C–B–A 兩種方向，分別處理 (a→b→c) 與 (b→a→c)
        for x, y in [(a, b), (b, a)]:
            # 3.3 取出 y 原子的所有鄰居 nb（除了 x 以外）
            for nb in mol.GetAtomWithIdx(y).GetNeighbors():
                c = nb.GetIdx()
                if c == x:
                    # 排除回頭的情況 (x→y→x)
                    continue
                # 3.4 根據配位數字典，組出 key 字串 "i-j-k"
                #     i = coord[x] (A 的配位數)、j = coord[y] (B)、k = coord[c] (C)
                key = f"{coord[x]}-{coord[y]}-{coord[c]}"
                
                # 3.5 在 Counter 裡對該 key 次數加 1
                counts[key] += 1

# 4. 統計完所有分子後，取出出現頻率最高的前 155 種 key
#    most_common(155) 會回傳一個 list of (key, count) 的 tuple
atom_keys = [k for k, _ in counts.most_common(155)]

def extract_atomic_fragments(mol):
    """
    提取原子級三原子片段指紋 (i-j-k)。
    輸入：RDKit Mol 物件
    輸出：一個長度為 len(atom_keys) 的 list，
          代表每種 i-j-k 組合的出現次數。
    """
    # 1. 計算每個原子的配位數 (coordination number)
    #    用原子索引當 key，鄰居原子數量當 value
    coord = {a.GetIdx(): len(a.GetNeighbors()) for a in mol.GetAtoms()}
    
    # 2. 初始化計數字典：對 atom_keys 中的每個 i-j-k 組合，初始值設為 0
    cnt = {k: 0 for k in atom_keys}
    
    # 3. 遍歷分子中所有的化學鍵 (bond)
    for bnd in mol.GetBonds():
        # 3.1 取得這條鍵連接的兩個原子索引 a, b
        a = bnd.GetBeginAtomIdx()
        b = bnd.GetEndAtomIdx()
        
        # 3.2 同時計算兩種方向：A→B→C 與 B→A→C
        for x, y in [(a, b), (b, a)]:
            # 3.3 取出中間原子 y 的所有鄰居原子
            for nb in mol.GetAtomWithIdx(y).GetNeighbors():
                c = nb.GetIdx()
                # 3.4 排除 A→B→A 的情況
                if c == x:
                    continue
                # 3.5 由三個原子的配位數組成 key 字串，格式 "i-j-k"
                key = f"{coord[x]}-{coord[y]}-{coord[c]}"
                
                # 3.6 如果這個 key 在我們關心的 atom_keys 清單中，就累加計數
                if key in cnt:
                    cnt[key] += 1
    
    # 4. 回傳固定順序的 list，順序對應 atom_keys
    return [cnt[k] for k in atom_keys]
