import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.MACCSkeys import GenMACCSKeys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# 讀入資料
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print("資料基本資訊:")
print(f"訓練集大小: {train.shape}")
print(f"測試集大小: {test.shape}")

# SMILES 轉 mol
train["mol"] = train["SMILES"].apply(Chem.MolFromSmiles)
test["mol"] = test["SMILES"].apply(Chem.MolFromSmiles)

# 四種單獨的指紋提取函數
def compute_morgan_fp(mol, radius=2, nBits=2048):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))

def compute_maccs_fp(mol):
    return np.array(GenMACCSKeys(mol))

def compute_avalon_fp(mol, nBits=512):
    return np.array(GetAvalonFP(mol, nBits=nBits))

def compute_rdk_fp(mol):
    return np.array(RDKFingerprint(mol))

# 計算四種指紋
fingerprints = {
    'Morgan': [compute_morgan_fp(mol) for mol in train["mol"]],
    'MACCS': [compute_maccs_fp(mol) for mol in train["mol"]],
    'Avalon': [compute_avalon_fp(mol) for mol in train["mol"]],
    'RDK': [compute_rdk_fp(mol) for mol in train["mol"]]
}

# 轉換為numpy陣列
X_fingerprints = {}
for name, fps in fingerprints.items():
    X_fingerprints[name] = np.array(fps)
    print(f"{name} 指紋維度: {X_fingerprints[name].shape}")

# 目標變數
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
y_data = train[targets].copy()

print(f"\n目標變數缺失值統計:")
for target in targets:
    missing = y_data[target].isnull().sum()
    available = y_data[target].notnull().sum()
    print(f"{target}: 缺失 {missing}, 可用 {available}")

# 建立性能比較矩陣
results = pd.DataFrame(index=targets, columns=list(fingerprints.keys()))
results_r2 = pd.DataFrame(index=targets, columns=list(fingerprints.keys()))

print(f"\n開始訓練模型 (4種指紋 × 5個目標變數 = 20個模型)...")

# 對每個目標變數和每種指紋組合建立模型
for target in targets:
    print(f"\n=== 處理目標變數: {target} ===")
    
    # 取得該目標的非空樣本
    mask = y_data[target].notnull()
    y_target = y_data[target][mask]
    
    print(f"可用樣本數: {len(y_target)}")
    
    for fp_name in fingerprints.keys():
        # 取得對應的指紋特徵
        X_target = X_fingerprints[fp_name][mask]
        
        # 切分訓練/驗證集
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )
        
        # 特徵標準化
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        # 訓練模型
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_tr_scaled, y_tr)
        
        # 預測與評估
        y_pred = model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # 儲存結果
        results.loc[target, fp_name] = f"{mse:.6f}"
        results_r2.loc[target, fp_name] = f"{r2:.4f}"
        
        print(f"  {fp_name:8s}: MSE={mse:.6f}, R²={r2:.4f}")

print(f"\n" + "="*60)
print("MSE 結果摘要 (越小越好):")
print(results)

print(f"\nR² 結果摘要 (越大越好):")
print(results_r2)

# 找出每個目標變數的最佳指紋
print(f"\n最佳指紋選擇 (基於 R²):")
for target in targets:
    r2_values = {fp: float(results_r2.loc[target, fp]) for fp in fingerprints.keys()}
    best_fp = max(r2_values, key=r2_values.get)
    best_r2 = r2_values[best_fp]
    print(f"{target:8s}: {best_fp} (R²={best_r2:.4f})")

# 儲存結果
results.to_csv("fingerprint_comparison_mse.csv")
results_r2.to_csv("fingerprint_comparison_r2.csv")
print(f"\n結果已儲存到 fingerprint_comparison_mse.csv 和 fingerprint_comparison_r2.csv")
