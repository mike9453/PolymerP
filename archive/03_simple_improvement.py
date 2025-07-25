#!/usr/bin/env python3
"""
簡化版階段1改進：組合指紋測試
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.MACCSkeys import GenMACCSKeys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("🧪 簡化版改進測試：最佳指紋組合")
print("=" * 50)

# 讀取資料
train = pd.read_csv("./data/train.csv")
train["mol"] = train["SMILES"].apply(Chem.MolFromSmiles)

# 指紋計算
def compute_morgan_fp(mol):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

def compute_maccs_fp(mol):
    return np.array(GenMACCSKeys(mol))

print("計算指紋...")
morgan_fps = np.array([compute_morgan_fp(mol) for mol in train["mol"]])
maccs_fps = np.array([compute_maccs_fp(mol) for mol in train["mol"]])

# 組合指紋
combined_fps = np.hstack([morgan_fps, maccs_fps])

print(f"Morgan 指紋: {morgan_fps.shape}")
print(f"MACCS 指紋: {maccs_fps.shape}")
print(f"組合指紋: {combined_fps.shape}")

# 測試目標
targets = ["FFV"]  # 先測試一個有最多數據的目標
y_data = train[targets].copy()

print(f"\n開始測試模型改進...")

# 基準模型 vs 改進模型
models = {
    'Baseline_RF': RandomForestRegressor(n_estimators=100, random_state=42),
    'Improved_RF': RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
}

# 特徵組合
feature_sets = {
    'Morgan_Only': morgan_fps,
    'MACCS_Only': maccs_fps,
    'Combined': combined_fps
}

results = []

for target in targets:
    print(f"\n=== 目標變數: {target} ===")
    
    # 取得非空樣本
    mask = y_data[target].notnull()
    y_target = y_data[target][mask]
    print(f"可用樣本數: {len(y_target)}")
    
    for feat_name, X_features in feature_sets.items():
        X_target = X_features[mask]
        
        for model_name, model in models.items():
            # 切分資料
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_target, y_target, test_size=0.2, random_state=42
            )
            
            # 標準化
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            # 訓練模型
            model.fit(X_tr_scaled, y_tr)
            y_pred = model.predict(X_val_scaled)
            r2 = r2_score(y_val, y_pred)
            
            results.append({
                'target': target,
                'features': feat_name,
                'model': model_name,
                'r2': r2
            })
            
            print(f"{feat_name:12s} + {model_name:12s}: R² = {r2:.4f}")

# 分析結果
print(f"\n" + "=" * 60)
print("📊 改進效果分析")
print("=" * 60)

# 基準線 (原始結果)
baseline_ffv = 0.6520  # 原始 Morgan + Basic RF 的結果

print(f"\n原始最佳結果 (FFV): {baseline_ffv:.4f}")
print("改進後結果:")

best_result = 0
best_config = ""

for result in results:
    if result['r2'] > best_result:
        best_result = result['r2']
        best_config = f"{result['features']} + {result['model']}"

improvement = best_result - baseline_ffv
improvement_pct = (improvement / baseline_ffv) * 100

print(f"最佳結果: {best_result:.4f} ({best_config})")
print(f"改進幅度: +{improvement:.4f} ({improvement_pct:+.1f}%)")

if improvement > 0:
    print("✅ 改進成功! 可以進入下一階段")
else:
    print("⚠️  改進效果不明顯，需要調整策略")

print(f"\n💡 下一步建議:")
print("1. 如果改進效果好 → 添加化學描述符")
print("2. 如果改進有限 → 嘗試其他算法 (XGBoost, LightGBM)")
print("3. 進階優化 → 超參數調優、交叉驗證")