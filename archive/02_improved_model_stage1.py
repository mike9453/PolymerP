#!/usr/bin/env python3
"""
階段1改進版本：組合指紋 + 改進的RandomForest
預期提升：3-8% R²
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, Descriptors, rdMolDescriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.MACCSkeys import GenMACCSKeys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 階段1改進版本：組合指紋 + 優化參數")
print("=" * 50)

# 讀入資料
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print(f"訓練集大小: {train.shape}")
print(f"測試集大小: {test.shape}")

# SMILES 轉 mol
train["mol"] = train["SMILES"].apply(Chem.MolFromSmiles)
test["mol"] = test["SMILES"].apply(Chem.MolFromSmiles)

# 指紋計算函數
def compute_morgan_fp(mol, radius=2, nBits=2048):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))

def compute_maccs_fp(mol):
    return np.array(GenMACCSKeys(mol))

def compute_avalon_fp(mol, nBits=512):
    return np.array(GetAvalonFP(mol, nBits=nBits))

def compute_rdk_fp(mol):
    return np.array(RDKFingerprint(mol))

# 新增：化學描述符函數
def compute_descriptors(mol):
    """計算重要的化學描述符"""
    desc = []
    desc.append(Descriptors.MolWt(mol))                    # 分子量
    desc.append(Descriptors.MolLogP(mol))                  # 疏水性
    desc.append(rdMolDescriptors.CalcTPSA(mol))            # 極性表面積
    desc.append(Descriptors.NumAromaticRings(mol))         # 芳香環數
    desc.append(rdMolDescriptors.CalcFractionCSP3(mol))    # sp3比例
    desc.append(Descriptors.NumRotatableBonds(mol))        # 可旋轉鍵
    desc.append(Descriptors.NumHDonors(mol))               # 氫鍵供體
    desc.append(Descriptors.NumHAcceptors(mol))            # 氫鍵受體
    desc.append(mol.GetNumHeavyAtoms())                    # 重原子數
    desc.append(Descriptors.NumRings(mol))                 # 環數
    return np.array(desc)

print("\n計算特徵...")

# 計算所有指紋
fingerprints = {}
fingerprints['Morgan'] = np.array([compute_morgan_fp(mol) for mol in train["mol"]])
fingerprints['MACCS'] = np.array([compute_maccs_fp(mol) for mol in train["mol"]])
fingerprints['Avalon'] = np.array([compute_avalon_fp(mol) for mol in train["mol"]])
fingerprints['RDK'] = np.array([compute_rdk_fp(mol) for mol in train["mol"]])

# 新增：計算化學描述符
descriptors = np.array([compute_descriptors(mol) for mol in train["mol"]])

print(f"Morgan 指紋: {fingerprints['Morgan'].shape}")
print(f"MACCS 指紋: {fingerprints['MACCS'].shape}")
print(f"Avalon 指紋: {fingerprints['Avalon'].shape}")
print(f"RDK 指紋: {fingerprints['RDK'].shape}")
print(f"化學描述符: {descriptors.shape}")

# 🔥 改進1：創建最佳組合特徵
def create_feature_combinations():
    """創建不同的特徵組合方案"""
    combinations = {}
    
    # 方案1：最佳單一指紋 (基於之前結果)
    combinations['Best_Single'] = {
        'Tg': fingerprints['MACCS'],
        'FFV': fingerprints['Morgan'], 
        'Tc': fingerprints['MACCS'],
        'Density': fingerprints['MACCS'],
        'Rg': fingerprints['Morgan']
    }
    
    # 方案2：最佳雙指紋組合
    morgan_maccs = np.hstack([fingerprints['Morgan'], fingerprints['MACCS']])
    combinations['Morgan_MACCS'] = {
        'Tg': morgan_maccs,
        'FFV': morgan_maccs,
        'Tc': morgan_maccs, 
        'Density': morgan_maccs,
        'Rg': morgan_maccs
    }
    
    # 方案3：指紋 + 描述符
    combinations['FP_Descriptors'] = {}
    for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        if target in ['FFV', 'Rg']:
            base_fp = fingerprints['Morgan']
        else:
            base_fp = fingerprints['MACCS']
        combinations['FP_Descriptors'][target] = np.hstack([base_fp, descriptors])
    
    return combinations

feature_combinations = create_feature_combinations()

# 🔥 改進2：優化的RandomForest參數
improved_models = {
    'Basic_RF': RandomForestRegressor(
        n_estimators=100, 
        random_state=42
    ),
    'Improved_RF': RandomForestRegressor(
        n_estimators=300,        # 增加樹數量
        max_depth=20,           # 限制深度
        min_samples_split=5,    # 分割最小樣本
        min_samples_leaf=2,     # 葉節點最小樣本
        max_features='sqrt',    # 隨機特徵數
        bootstrap=True,         # 自助採樣
        random_state=42,
        n_jobs=-1              # 並行計算
    )
}

# 目標變數
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
y_data = train[targets].copy()

print(f"\n開始模型比較...")
print("=" * 80)

# 結果儲存
all_results = []

# 測試所有組合
for combo_name, combo_features in feature_combinations.items():
    print(f"\n🧪 測試特徵組合: {combo_name}")
    print("-" * 50)
    
    for model_name, model in improved_models.items():
        combo_results = {'combination': combo_name, 'model': model_name}
        
        for target in targets:
            # 取得該目標的非空樣本
            mask = y_data[target].notnull()
            y_target = y_data[target][mask]
            X_target = combo_features[target][mask]
            
            # 切分資料
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_target, y_target, test_size=0.2, random_state=42
            )
            
            # 特徵標準化
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            # 訓練模型
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_tr_scaled, y_tr)
            
            # 預測與評估
            y_pred = model_copy.predict(X_val_scaled)
            r2 = r2_score(y_val, y_pred)
            
            combo_results[target] = f"{r2:.4f}"
            
        all_results.append(combo_results)
        
        # 顯示結果
        print(f"{model_name:12s}:", end="")
        for target in targets:
            print(f" {target}={combo_results[target]}", end="")
        print()

# 轉換為DataFrame並儲存
results_df = pd.DataFrame(all_results)
results_df.to_csv("stage1_improvement_results.csv", index=False)

print(f"\n" + "=" * 80)
print("📊 階段1改進結果摘要")
print("=" * 80)

# 找出每個目標的最佳組合
print("\n🏆 最佳結果比較:")
print("-" * 50)

# 讀取原始結果作為基準
baseline_results = {
    'Tg': 0.4573, 'FFV': 0.6520, 'Tc': 0.7046, 
    'Density': 0.7401, 'Rg': 0.6705
}

for target in targets:
    best_result = 0
    best_combo = ""
    best_model = ""
    
    for result in all_results:
        current_r2 = float(result[target])
        if current_r2 > best_result:
            best_result = current_r2
            best_combo = result['combination']
            best_model = result['model']
    
    baseline = baseline_results[target]
    improvement = best_result - baseline
    improvement_pct = (improvement / baseline) * 100
    
    print(f"{target:8s}: {baseline:.4f} → {best_result:.4f} "
          f"(+{improvement:+.4f}, {improvement_pct:+.1f}%) "
          f"[{best_combo}, {best_model}]")

print(f"\n💾 詳細結果已儲存至: stage1_improvement_results.csv")
print(f"🎯 下一步建議: 如果改進效果良好，可以進入階段2 (添加更多算法)")