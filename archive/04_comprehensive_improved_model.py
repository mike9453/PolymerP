#!/usr/bin/env python3
"""
完整改進版本：
1. 擴展所有目標變數 (Tg, FFV, Tc, Density, Rg)
2. 添加完整分子描述符
3. 組合最佳指紋 (Morgan + MACCS)
4. 交叉驗證評估
5. 完整評估指標
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 完整改進版聚合物性質預測模型")
print("=" * 60)
print("✅ 擴展所有目標變數")
print("✅ 組合指紋 (Morgan + MACCS)")
print("✅ 完整分子描述符")
print("✅ 交叉驗證評估")
print("✅ 完整評估指標")
print("=" * 60)

# ===============================
# 1. 資料載入與基本處理
# ===============================
print("\n📁 載入資料...")
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print(f"訓練集大小: {train.shape}")
print(f"測試集大小: {test.shape}")

# SMILES 轉分子
train["mol"] = train["SMILES"].apply(Chem.MolFromSmiles)
test["mol"] = test["SMILES"].apply(Chem.MolFromSmiles)

# ===============================
# 2. 特徵提取函數定義
# ===============================
print("\n🧬 定義特徵提取函數...")

def compute_morgan_fp(mol, radius=2, nBits=2048):
    """計算Morgan指紋"""
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))

def compute_maccs_fp(mol):
    """計算MACCS指紋"""
    return np.array(GenMACCSKeys(mol))

def compute_comprehensive_descriptors(mol):
    """計算完整的分子描述符集合"""
    desc = []
    
    # ===================
    # 基本物理化學性質
    # ===================
    desc.append(Descriptors.MolWt(mol))                    # 分子量
    desc.append(Descriptors.ExactMolWt(mol))               # 精確分子量
    desc.append(mol.GetNumHeavyAtoms())                    # 重原子數
    desc.append(mol.GetNumAtoms())                         # 總原子數
    desc.append(Descriptors.NumValenceElectrons(mol))      # 價電子數
    
    # ===================
    # 疏水性與極性
    # ===================
    desc.append(Descriptors.MolLogP(mol))                  # 疏水性係數
    desc.append(Descriptors.MolMR(mol))                    # 摩爾折射率
    desc.append(rdMolDescriptors.CalcTPSA(mol))            # 拓撲極性表面積
    
    # ===================
    # 氫鍵特性
    # ===================
    desc.append(Descriptors.NumHDonors(mol))               # 氫鍵供體數
    desc.append(Descriptors.NumHAcceptors(mol))            # 氫鍵受體數
    
    # ===================
    # 結構柔性
    # ===================
    desc.append(Descriptors.NumRotatableBonds(mol))        # 可旋轉鍵數
    desc.append(rdMolDescriptors.CalcFractionCSP3(mol))    # sp3碳比例
    
    # ===================
    # 環系統特徵
    # ===================
    desc.append(Descriptors.NumRings(mol))                 # 環總數
    desc.append(Descriptors.NumAromaticRings(mol))         # 芳香環數
    desc.append(Descriptors.NumSaturatedRings(mol))        # 飽和環數
    desc.append(Descriptors.NumAliphaticRings(mol))        # 脂肪環數
    
    # ===================
    # 分子複雜度
    # ===================
    desc.append(Descriptors.BertzCT(mol))                  # Bertz複雜度指數
    desc.append(Descriptors.HallKierAlpha(mol))            # Hall-Kier Alpha指數
    desc.append(Descriptors.BalabanJ(mol))                 # Balaban J指數
    
    # ===================
    # 連通性指數 (Chi indices)
    # ===================
    desc.append(Descriptors.Chi0(mol))                     # Chi0 連通性指數
    desc.append(Descriptors.Chi0n(mol))                    # Chi0n
    desc.append(Descriptors.Chi0v(mol))                    # Chi0v
    desc.append(Descriptors.Chi1(mol))                     # Chi1
    desc.append(Descriptors.Chi1n(mol))                    # Chi1n
    desc.append(Descriptors.Chi1v(mol))                    # Chi1v
    
    # ===================
    # Kappa 形狀指數
    # ===================
    desc.append(Descriptors.Kappa1(mol))                   # Kappa1
    desc.append(Descriptors.Kappa2(mol))                   # Kappa2
    desc.append(Descriptors.Kappa3(mol))                   # Kappa3
    
    # ===================
    # 聚合物特有描述符
    # ===================
    desc.append(Descriptors.FractionCarbons(mol))          # 碳原子比例
    desc.append(Descriptors.NumHeteroatoms(mol))           # 雜原子數
    desc.append(mol.GetNumBonds())                         # 鍵數
    
    # 處理可能的NaN值
    desc = [0.0 if (pd.isna(x) or np.isinf(x)) else float(x) for x in desc]
    
    return np.array(desc)

# ===============================
# 3. 計算所有特徵
# ===============================
print("\n⚙️  計算特徵矩陣...")

# 指紋特徵
print("  計算Morgan指紋...")
morgan_fps = np.array([compute_morgan_fp(mol) for mol in train["mol"]])

print("  計算MACCS指紋...")
maccs_fps = np.array([compute_maccs_fp(mol) for mol in train["mol"]])

print("  計算分子描述符...")
descriptors = np.array([compute_comprehensive_descriptors(mol) for mol in train["mol"]])

# 組合所有特徵
print("  組合特徵...")
X_combined = np.hstack([
    morgan_fps,     # 2048維
    maccs_fps,      # 167維
    descriptors     # 28維
])

print(f"✅ 最終特徵矩陣維度: {X_combined.shape}")
print(f"   - Morgan指紋: {morgan_fps.shape[1]} 維")
print(f"   - MACCS指紋: {maccs_fps.shape[1]} 維") 
print(f"   - 分子描述符: {descriptors.shape[1]} 維")
print(f"   - 總計: {X_combined.shape[1]} 維")

# ===============================
# 4. 目標變數準備
# ===============================
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
y_data = train[targets].copy()

print(f"\n📊 目標變數統計:")
for target in targets:
    missing = y_data[target].isnull().sum()
    available = y_data[target].notnull().sum()
    percentage = (available / len(y_data)) * 100
    print(f"  {target:8s}: 可用 {available:4d} ({percentage:5.1f}%), 缺失 {missing:4d}")

# ===============================
# 5. 模型訓練與評估
# ===============================
print(f"\n🤖 開始訓練模型...")
print("=" * 60)

# 改進的RandomForest參數
improved_rf = RandomForestRegressor(
    n_estimators=300,        # 增加樹數量提升性能
    max_depth=20,           # 限制深度防止過擬合
    min_samples_split=5,    # 最小分割樣本數
    min_samples_leaf=2,     # 葉節點最小樣本數
    max_features='sqrt',    # 隨機特徵選擇
    bootstrap=True,         # 自助採樓
    random_state=42,
    n_jobs=-1              # 並行計算
)

# 結果儲存
results = []
detailed_results = {}

# 交叉驗證設定
cv_folds = 5
cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

for target in targets:
    print(f"\n🎯 處理目標變數: {target}")
    print("-" * 40)
    
    # 取得該目標的非空樣本
    mask = y_data[target].notnull()
    X_target = X_combined[mask]
    y_target = y_data[target][mask]
    
    n_samples = len(y_target)
    print(f"可用樣本數: {n_samples}")
    
    if n_samples < 50:  # 樣本太少跳過交叉驗證
        print("⚠️  樣本數太少，跳過該目標變數")
        continue
    
    # 特徵標準化
    scaler = StandardScaler()
    X_target_scaled = scaler.fit_transform(X_target)
    
    # =========================
    # Hold-out驗證
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_target_scaled, y_target, test_size=0.2, random_state=42
    )
    
    # 訓練模型
    model = RandomForestRegressor(**improved_rf.get_params())
    model.fit(X_train, y_train)
    
    # 預測
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # =========================
    # 交叉驗證
    # =========================
    print("  執行交叉驗證...")
    cv_r2_scores = cross_val_score(model, X_target_scaled, y_target, 
                                  cv=cv, scoring='r2', n_jobs=-1)
    cv_neg_mse_scores = cross_val_score(model, X_target_scaled, y_target, 
                                       cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_neg_mae_scores = cross_val_score(model, X_target_scaled, y_target, 
                                       cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # =========================
    # 計算所有評估指標
    # =========================
    
    # Hold-out指標
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # 交叉驗證指標
    cv_r2_mean = cv_r2_scores.mean()
    cv_r2_std = cv_r2_scores.std()
    cv_mse_mean = -cv_neg_mse_scores.mean()
    cv_mse_std = cv_neg_mse_scores.std()
    cv_rmse_mean = np.sqrt(cv_mse_mean)
    cv_mae_mean = -cv_neg_mae_scores.mean()
    cv_mae_std = cv_neg_mae_scores.std()
    
    # 相對指標
    y_std = y_target.std()
    relative_rmse = test_rmse / y_std
    relative_mae = test_mae / y_std
    
    # 過擬合檢查
    overfitting = train_r2 - test_r2
    
    # 儲存結果
    result = {
        'target': target,
        'n_samples': n_samples,
        
        # Hold-out驗證結果
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        
        # 交叉驗證結果
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'cv_mse_mean': cv_mse_mean,
        'cv_mse_std': cv_mse_std,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_mae_mean': cv_mae_mean,
        'cv_mae_std': cv_mae_std,
        
        # 相對指標
        'target_std': y_std,
        'relative_rmse': relative_rmse,
        'relative_mae': relative_mae,
        
        # 模型診斷
        'overfitting': overfitting,
    }
    
    results.append(result)
    detailed_results[target] = result
    
    # 顯示關鍵指標
    print(f"  Hold-out R²: {test_r2:.4f}")
    print(f"  交叉驗證 R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"  相對RMSE: {relative_rmse:.4f}")
    print(f"  過擬合程度: {overfitting:.4f}")

# ===============================
# 6. 結果分析與評估
# ===============================
print(f"\n" + "=" * 80)
print("📊 完整模型評估報告")
print("=" * 80)

# 轉換為DataFrame便於分析
results_df = pd.DataFrame(results)

print(f"\n🏆 模型性能總覽")
print("-" * 50)
print(f"{'目標':8s} {'樣本數':>6s} {'Hold-out R²':>12s} {'CV R²':>15s} {'相對RMSE':>10s} {'過擬合':>8s}")
print("-" * 50)

for _, row in results_df.iterrows():
    print(f"{row['target']:8s} "
          f"{row['n_samples']:6.0f} "
          f"{row['test_r2']:12.4f} "
          f"{row['cv_r2_mean']:8.4f}±{row['cv_r2_std']:5.3f} "
          f"{row['relative_rmse']:10.4f} "
          f"{row['overfitting']:8.4f}")

# 基準結果比較 (來自原始模型)
baseline_results = {
    'Tg': 0.4573, 'FFV': 0.6520, 'Tc': 0.7046, 
    'Density': 0.7401, 'Rg': 0.6705
}

print(f"\n📈 與基準模型比較")
print("-" * 50)
print(f"{'目標':8s} {'基準R²':>8s} {'改進R²':>8s} {'提升':>8s} {'提升%':>8s}")
print("-" * 50)

total_improvement = 0
valid_comparisons = 0

for _, row in results_df.iterrows():
    target = row['target']
    if target in baseline_results:
        baseline = baseline_results[target]
        improved = row['test_r2']
        improvement = improved - baseline
        improvement_pct = (improvement / baseline) * 100
        
        total_improvement += improvement_pct
        valid_comparisons += 1
        
        print(f"{target:8s} "
              f"{baseline:8.4f} "
              f"{improved:8.4f} "
              f"{improvement:+8.4f} "
              f"{improvement_pct:+7.1f}%")

if valid_comparisons > 0:
    avg_improvement = total_improvement / valid_comparisons
    print(f"{'平均':8s} {'':8s} {'':8s} {'':8s} {avg_improvement:+7.1f}%")

print(f"\n🔍 模型品質評估")
print("-" * 50)

# 品質等級分類
def classify_performance(r2, relative_rmse, overfitting):
    """根據多個指標分類模型品質"""
    if r2 >= 0.8 and relative_rmse <= 0.3 and abs(overfitting) <= 0.05:
        return "優秀"
    elif r2 >= 0.6 and relative_rmse <= 0.5 and abs(overfitting) <= 0.1:
        return "良好"
    elif r2 >= 0.4 and relative_rmse <= 0.7 and abs(overfitting) <= 0.15:
        return "可接受"
    else:
        return "需改進"

print(f"{'目標':8s} {'R²等級':>8s} {'RMSE等級':>10s} {'過擬合':>8s} {'總評':>8s}")
print("-" * 50)

for _, row in results_df.iterrows():
    r2_level = "優秀" if row['test_r2'] >= 0.8 else "良好" if row['test_r2'] >= 0.6 else "可接受" if row['test_r2'] >= 0.4 else "差"
    rmse_level = "優秀" if row['relative_rmse'] <= 0.3 else "良好" if row['relative_rmse'] <= 0.5 else "可接受" if row['relative_rmse'] <= 0.7 else "差"
    overfitting_status = "正常" if abs(row['overfitting']) <= 0.1 else "過擬合" if row['overfitting'] > 0.1 else "欠擬合"
    overall = classify_performance(row['test_r2'], row['relative_rmse'], row['overfitting'])
    
    print(f"{row['target']:8s} "
          f"{r2_level:>8s} "
          f"{rmse_level:>10s} "
          f"{overfitting_status:>8s} "
          f"{overall:>8s}")

print(f"\n💡 建議與下一步")
print("-" * 50)

# 自動生成建議
best_target = results_df.loc[results_df['test_r2'].idxmax(), 'target']
worst_target = results_df.loc[results_df['test_r2'].idxmin(), 'target']

print(f"✅ 表現最佳: {best_target} (R² = {results_df['test_r2'].max():.4f})")
print(f"⚠️  需要改進: {worst_target} (R² = {results_df['test_r2'].min():.4f})")

# 根據結果給出具體建議
avg_r2 = results_df['test_r2'].mean()
if avg_r2 >= 0.7:
    print("🎉 模型整體表現優秀！建議：")
    print("   - 可以嘗試集成學習進一步提升")
    print("   - 考慮超參數精調")
elif avg_r2 >= 0.5:
    print("👍 模型表現良好，建議：")
    print("   - 嘗試XGBoost/LightGBM算法")
    print("   - 添加更多領域特徵")
    print("   - 考慮特徵選擇")
else:
    print("🔧 模型需要改進，建議：")
    print("   - 檢查特徵工程")
    print("   - 嘗試不同算法")
    print("   - 增加數據量")

# 過擬合診斷
overfitting_targets = results_df[results_df['overfitting'] > 0.1]['target'].tolist()
if overfitting_targets:
    print(f"⚠️  發現過擬合: {', '.join(overfitting_targets)}")
    print("   - 建議減少模型複雜度或增加正則化")

# 儲存結果
results_df.to_csv("comprehensive_model_results.csv", index=False)
print(f"\n💾 詳細結果已儲存至: comprehensive_model_results.csv")

print(f"\n" + "=" * 80)
print("🎯 模型改進完成！")
print("=" * 80)