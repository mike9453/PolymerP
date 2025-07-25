#!/usr/bin/env python3
"""
演示分子指紋 vs 化學描述符的差異
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import numpy as np

# 示例分子：苯甲酸
smiles = "c1ccc(cc1)C(=O)O"
mol = Chem.MolFromSmiles(smiles)

print("分子：苯甲酸 (Benzoic acid)")
print(f"SMILES: {smiles}")
print("=" * 50)

# 1. 分子指紋 (二進制)
print("\n🧬 分子指紋 (Molecular Fingerprints):")
print("-" * 30)

# Morgan 指紋
morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=64)  # 簡化為64位展示
morgan_bits = np.array(morgan_fp)
print(f"Morgan 指紋 (64位):")
print(f"前20位: {morgan_bits[:20]}")
print(f"啟動的位數: {np.sum(morgan_bits)}/64")

# 2. 化學描述符 (數值)
print("\n📊 化學描述符 (Chemical Descriptors):")
print("-" * 30)

descriptors = {
    '分子量 (MW)': f"{Descriptors.MolWt(mol):.2f} g/mol",
    '親脂性 (LogP)': f"{Descriptors.MolLogP(mol):.2f}",
    '極性表面積 (TPSA)': f"{rdMolDescriptors.CalcTPSA(mol):.2f} Ų", 
    '氫鍵供體': f"{Descriptors.NumHDonors(mol)}",
    '氫鍵受體': f"{Descriptors.NumHAcceptors(mol)}",
    '可旋轉鍵': f"{Descriptors.NumRotatableBonds(mol)}",
    '芳香環數': f"{Descriptors.NumAromaticRings(mol)}",
    'sp3碳比例': f"{rdMolDescriptors.CalcFractionCSP3(mol):.2f}",
    '重原子數': f"{mol.GetNumHeavyAtoms()}",
    '總原子數': f"{mol.GetNumAtoms()}"
}

for name, value in descriptors.items():
    print(f"{name:15s}: {value}")

print("\n🔍 關鍵差異:")
print("-" * 30)
print("指紋告訴我們: 這個分子'像什麼'（結構模式）")
print("描述符告訴我們: 這個分子'是什麼'（具體性質）")

print("\n💡 在機器學習中:")
print("-" * 30)
print("• 指紋: 高維特徵，適合捕捉複雜結構模式")
print("• 描述符: 低維特徵，每個都有明確物理意義")
print("• 組合使用: 既有結構信息又有性質信息，通常效果最好！")

# 3. 組合特徵示例
print(f"\n🤝 組合特徵維度:")
print("-" * 30)
morgan_2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
desc_values = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
              rdMolDescriptors.CalcTPSA(mol), Descriptors.NumHDonors(mol)]

print(f"Morgan指紋: {len(morgan_2048)} 維")
print(f"選擇的描述符: {len(desc_values)} 維") 
print(f"組合後總維度: {len(morgan_2048) + len(desc_values)} 維")