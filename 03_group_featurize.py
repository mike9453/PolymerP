import numpy as np
from rdkit import Chem
import pandas as pd

# --------------------------------------------------
# 1. 準備 197 種 block-level SMARTS（需自行補齊）
# --------------------------------------------------
raw_smarts = [
    # 1. 碳主體
    "[CX4]",                         # sp3 烷基碳
    "[CX3]=[CX3]",                  # sp2 烯烃碳
    "[CX2]#[CX2]",                  # sp 炔烃碳
    "c",                             # 芳香碳
    "[CX3]=[CX2]=[CX3]",            # 烯炔總稱 (Allene)
    "[CX4H3][#6]",                  # 伯碳
    "[CX4H2]([#6])[#6]",            # 仲碳
    "[CX4H1]([#6])([#6])[#6]",      # 叔碳
    "[CX4]([#6])([#6])([#6])[#6]",  # 季碳

    # 2. 羰基及衍生物
    "[CX3]=[OX1]",                  # 羰基
    "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]", # 共振羰基
    "[CX3](=[OX1])C",               # 羰基與碳相連
    "[OX1]=CN",                     # 羰基與氮相連
    "[CX3](=[OX1])O",               # 羰基與氧相連
    "[CX3](=[OX1])[OX2][CX3](=[OX1])", # 酸酐
    "[CX3](=[OX1])[F,Cl,Br,I]",     # 酰卤化物
    "[CX3H1](=O)[#6]",              # 醛
    "[#6][CX3](=O)[#6]",            # 酮
    "[CX3](=O)[OX2H1]",             # 羧酸
    "[CX3](=O)[OX1H0-,OX2H1]",      # 羧酸或共軛鹼
    "[CX3](=O)[OX2H0][#6]",         # 酯
    "[NX3][CX3](=[OX1])[#6]",       # 酰胺
    "[NX3][CX3]=[NX3+]",            # 酰胺鎓 (amidinium)
    "[CX3](=O)[O-]",                # 羧酸鹽
    "[CX3](=[OX1])(O)O",            # 碳酸或碳酸酯
    "C[OX2][CX3](=[OX1])[OX2]C",    # 碳酸二酯

    # 2.2 RDKit/補充羰基衍生物
    "[$(S-!@[#6])](=O)(=O)(Cl)",   # 磺酰氯
    "[$(B-!@[#6])](O)(O)",          # 硼酸
    "[$(N-!@[#6])](=!@C=!@O)",      # 異氰酸酯
    "[CH;D2;!$(C-[!#6;!#1])]=O",    # RDKit 醛定義
    "[F,Cl,Br,I]-!@[#6]",           # 通用鹵代烷
    "[N;H0;$(N-[#6]);D3](=[O;D1])~[O;D1]",   # 硝基 (精確版)
    "[N;H0;$(N-[#6]);D2]=[N;D2]=[N;D1]",     # 叠氮 (精確版)
    "[C;$(C#[CH])]",                # 末端炔基

    # 3. 含氧官能團
    "[OX2H][#6]",                   # 醇
    "[OX2H][CX4H2]",               # 伯醇
    "[OX2H][CX4H]",                # 仲醇
    "[OX2H][CX4D4]",               # 叔醇
    "[OD2]([#6])[#6]",              # 醚
    "[OX2]([CX4])[CX4]",           # 二烷基醚
    "[OX2](c)[CX4]",                # 烷基‑芳香醚
    "[OX2H][c]",                    # 酚
    "[OX2H][$(C=C),$(cc)]",        # 酚或烯醇
    "[OX2,OX1-][OX2,OX1-]",         # 過氧化物
    "[OH]-*=[!#6]",                 # 酸性羥基

    # 4. 含氮官能團
    "[NX3;H2,H1;!$(NC=O)]",         # 一級/二級胺
    "[N;$(N-[#6]);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]", # 通用胺
    "[N;H2;D1]",                    # 伯胺
    "[N;H1;D2]",                    # 仲胺
    "[N;H0;D3]",                    # 叔胺
    "[CX3]=[NX2]",                 # 亞胺
    "[NX3+]=[CX3]",                # 亞胺鎓
    "[NX3][OX1H]",                 # 羥胺
    "[NX1]#[CX2]",                 # 腈
    "[CX1-]#[NX2+]",               # 異腈
    "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]", # 通用硝基
    "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]", # 通用叠氮
    "[NX2]=[OX1]",                  # 亞硝基
    "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",   # 氨基甲酸酯/羧胺
    "[NX3][CX3](=[OX1])[OX2H0]",   # 氨基甲酸酯 (酯)
    "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]", # 氨基甲酸酸
    "[NX3][CX3]=[NX3+]",           # amidinium
    "[$(N-!@[#6])](=!@C=!@S)",      # 異硫氰酸酯

    # 5. 含硫官能團
    "[SX2H]",                      # 硫醇
    "[SX2H,SX1H0-]",               # 硫醇/硫醇鹽
    "[SX2H0][!#16]",              # 硫化物
    "[SX2]([CX4])[CX4]",          # 二烷基硫醚
    "[SX2H0][SX2H0]",             # 二硫化物
    "[SX2D2][SX2D2]",             # 二硫化物 (雙鍵版本)
    "[NX3][CX3]=[SX1]",           # 硫酰胺
    "[S](=O)(=O)(Cl)",             # 磺酰氯
    "[SX4](=O)(=O)([#6])[OX2H]",  # 磺酸
    "[SX4]([NX3])(=O)(=O)[#6]",   # 磺酰胺
    "[SX3]=[OX1]",                # 亞砜 (低特異性)
    "[SX3](=[OX1])([#6])[#6]",    # 亞砜 (高特異性)

    # 6. 含磷官能團
    "[P](=O)([O][H0,OH])([O][H0,OH])([O][H0,OH])", # 磷酸
    "[P](=O)([OX2][#6])([O][H0,OH])([O][H0,OH])",  # 磷酸酯
    "[P+]([OX1-])([OX2][#6])([O][H0,OH])([O][H0,OH])", # 磷酸酯陰離子

    # 7. 含鹵素官能團
    "[F,Cl,Br,I]",               # 任意鹵素
    "[F,Cl,Br,I]-!@[#6]",        # 鹵素取代烷碳
    "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]", # 三鹵素
    "[ClX1][CX4]",               # 烷基氯
    "[FX1][CX4]",                # 烷基氟
    "[BrX1][CX4]",               # 烷基溴
    "[IX1][CX4]",                # 烷基碘

    # 8. 酸鹼特徵與氫鍵
    "[CX3](=O)[OX2H,OX1H0-]",    # 酸/共軛鹼組合
    "+1~*~*~[-1]",               # 內鹽 (zwitterion)
    "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]", # 氫鍵受體
    "[!$([#6,H0,-,-2,-3])]",     # 氫鍵供體 (通用)
    "[!H0;#7,#8,#9]",            # 氫鍵供體 (N/O/F)

    # 9. 環與拓樸特徵
    "[cX3](:*):*",               # 芳香 sp2 碳在環中
    "*-!:aa-!:*",                # 鄰位取代
    "*-!:aaa-!:*",               # 間位取代
    "*-!:aaaa-!:*",              # 對位取代
    "*!@*",                      # 非環單鍵
    "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]", # 長鏈 (>8)
    "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]", # 螺環中心
]


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
