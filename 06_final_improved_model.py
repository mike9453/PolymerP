#!/usr/bin/env python3
"""
最終改進版本：基於01_load_inspect.py架構
✅ 擴展所有目標變數
✅ 組合指紋 + 分子描述符
✅ 交叉驗證
✅ 完整評估指標
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 最終改進版聚合物性質預測模型")
print("=" * 50)
print("基於 01_load_inspect.py 架構的完整改進")

# 讀入資料
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print(f"訓練集大小: {train.shape}")
print(f"測試集大小: {test.shape}")

# SMILES 轉 mol
train["mol"] = train["SMILES"].apply(Chem.MolFromSmiles)
test["mol"] = test["SMILES"].apply(Chem.MolFromSmiles)

# 改進的指紋組合函數
def compute_enhanced_fingerprints(mol):
    """組合Morgan + MACCS指紋 + 關鍵描述符"""
    # 指紋部分
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp2 = GenMACCSKeys(mol)
    
    # 關鍵分子描述符
    descriptors = [
        Descriptors.MolWt(mol),                    # 分子量 - 影響所有物性
        Descriptors.MolLogP(mol),                  # 疏水性 - 影響FFV, Density
        rdMolDescriptors.CalcTPSA(mol),            # 極性表面積 - 影響結晶性
        Descriptors.NumAromaticRings(mol),         # 芳香環 - 影響Tg剛性
        rdMolDescriptors.CalcFractionCSP3(mol),    # sp3比例 - 影響柔性
        Descriptors.NumRotatableBonds(mol),        # 可旋轉鍵 - 影響Rg
        Descriptors.NumHDonors(mol),               # 氫鍵供體 - 影響Tc
        Descriptors.NumHAcceptors(mol),            # 氫鍵受體 - 影響結晶
        mol.GetNumHeavyAtoms(),                    # 分子大小 - 影響密度
        Descriptors.NumRings(mol),                 # 環數 - 影響剛性
    ]
    
    # 處理可能的異常值
    descriptors = [0.0 if (pd.isna(x) or np.isinf(x)) else float(x) for x in descriptors]
    
    # 組合所有特徵
    return np.concatenate([np.array(fp1), np.array(fp2), np.array(descriptors)])

print("\n計算增強型特徵...")
X_train = np.array([compute_enhanced_fingerprints(mol) for mol in train["mol"]])
X_test = np.array([compute_enhanced_fingerprints(mol) for mol in test["mol"]])

print(f"特徵維度: {X_train.shape[1]} (Morgan:2048 + MACCS:167 + Descriptors:10)")

# 目標變數
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
y_data = train[targets].copy()

print(f"\n目標變數統計:")
for target in targets:
    missing = y_data[target].isnull().sum()
    available = y_data[target].notnull().sum()
    percentage = (available / len(y_data)) * 100
    print(f"  {target}: 可用 {available} ({percentage:.1f}%), 缺失 {missing}")

# 改進的模型參數
improved_model = RandomForestRegressor(
    n_estimators=200,        # 平衡性能與速度
    max_depth=15,           # 防止過擬合
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# 基準結果 (來自原始模型)
baseline_results = {
    'Tg': 0.4573, 'FFV': 0.6520, 'Tc': 0.7046, 
    'Density': 0.7401, 'Rg': 0.6705
}

print(f"\n開始訓練與評估...")
print("=" * 60)

# 結果儲存
comprehensive_results = []

for target in targets:
    print(f"\n🎯 目標變數: {target}")
    print("-" * 30)
    
    # 取得該目標的非空樣本
    mask = y_data[target].notnull()
    X_target = X_train[mask]
    y_target = y_data[target][mask]
    
    n_samples = len(y_target)
    print(f"可用樣本數: {n_samples}")
    
    if n_samples < 50:
        print("⚠️  樣本數太少，跳過")
        continue
    
    # 切分訓練集與驗證集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_target, y_target, test_size=0.2, random_state=42
    )
    
    # 特徵標準化
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    
    # 訓練模型
    model = RandomForestRegressor(**improved_model.get_params())
    model.fit(X_tr_scaled, y_tr)
    
    # 預測
    y_pred_tr = model.predict(X_tr_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    # Hold-out評估指標
    train_r2 = r2_score(y_tr, y_pred_tr)
    val_r2 = r2_score(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    
    # 相對誤差
    y_std = y_target.std()
    relative_rmse = val_rmse / y_std
    
    # 過擬合檢查
    overfitting = train_r2 - val_r2
    
    # 交叉驗證
    print("  執行交叉驗證...")
    X_all_scaled = scaler.fit_transform(X_target)
    cv_r2_scores = cross_val_score(model, X_all_scaled, y_target, cv=5, scoring='r2')
    cv_mse_scores = cross_val_score(model, X_all_scaled, y_target, cv=5, scoring='neg_mean_squared_error')
    
    cv_r2_mean = cv_r2_scores.mean()
    cv_r2_std = cv_r2_scores.std()
    cv_rmse_mean = np.sqrt(-cv_mse_scores.mean())
    
    # 與基準比較
    baseline = baseline_results.get(target, 0)
    improvement = val_r2 - baseline
    improvement_pct = (improvement / baseline * 100) if baseline > 0 else 0
    
    # 儲存結果
    result = {
        'target': target,
        'n_samples': n_samples,
        'baseline_r2': baseline,
        'holdout_r2': val_r2,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'rmse': val_rmse,
        'cv_rmse': cv_rmse_mean,
        'relative_rmse': relative_rmse,
        'overfitting': overfitting,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }
    comprehensive_results.append(result)
    
    # 顯示結果
    print(f"  Hold-out R²: {val_r2:.4f}")
    print(f"  交叉驗證 R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"  RMSE: {val_rmse:.4f} (相對: {relative_rmse:.3f})")
    print(f"  與基準比較: {baseline:.4f} → {val_r2:.4f} ({improvement_pct:+.1f}%)")
    print(f"  過擬合程度: {overfitting:.4f}")

# ===============================
# 綜合結果分析
# ===============================
print(f"\n" + "=" * 70)
print("📊 完整模型評估報告")
print("=" * 70)

# 轉換為DataFrame
results_df = pd.DataFrame(comprehensive_results)

print(f"\n🏆 性能摘要")
print("-" * 50)
print(f"{'目標':8s} {'樣本數':>6s} {'基準R²':>8s} {'改進R²':>8s} {'CV R²':>12s} {'提升%':>8s}")
print("-" * 50)

total_improvement = 0
count = 0

for _, row in results_df.iterrows():
    print(f"{row['target']:8s} "
          f"{row['n_samples']:6.0f} "
          f"{row['baseline_r2']:8.4f} "
          f"{row['holdout_r2']:8.4f} "
          f"{row['cv_r2_mean']:6.4f}±{row['cv_r2_std']:4.3f} "
          f"{row['improvement_pct']:+7.1f}%")
    
    if row['improvement_pct'] != 0:
        total_improvement += row['improvement_pct']
        count += 1

if count > 0:
    avg_improvement = total_improvement / count
    print(f"{'平均':8s} {'':6s} {'':8s} {'':8s} {'':12s} {avg_improvement:+7.1f}%")

print(f"\n🔍 模型品質分析")
print("-" * 50)

def get_quality_rating(r2, rel_rmse, overfitting):
    """綜合品質評分"""
    if r2 >= 0.8 and rel_rmse <= 0.3 and abs(overfitting) <= 0.05:
        return "優秀", "🏆"
    elif r2 >= 0.6 and rel_rmse <= 0.5 and abs(overfitting) <= 0.1:
        return "良好", "✅"
    elif r2 >= 0.4 and rel_rmse <= 0.7 and abs(overfitting) <= 0.15:
        return "可接受", "⚠️"
    else:
        return "需改進", "❌"

print(f"{'目標':8s} {'R²':>8s} {'相對RMSE':>10s} {'過擬合':>8s} {'評級':>6s}")
print("-" * 50)

for _, row in results_df.iterrows():
    quality, emoji = get_quality_rating(row['holdout_r2'], row['relative_rmse'], row['overfitting'])
    print(f"{row['target']:8s} "
          f"{row['holdout_r2']:8.4f} "
          f"{row['relative_rmse']:10.4f} "
          f"{row['overfitting']:8.4f} "
          f"{emoji} {quality}")

# 特徵重要性分析 (針對表現最好的目標)
best_target_idx = results_df['holdout_r2'].idxmax()
best_target = results_df.iloc[best_target_idx]['target']

print(f"\n🔬 模型診斷")
print("-" * 50)
print(f"表現最佳: {best_target} (R² = {results_df.iloc[best_target_idx]['holdout_r2']:.4f})")

# 穩健性分析
cv_stability = results_df['cv_r2_std'].mean()
print(f"交叉驗證穩定性: {cv_stability:.4f} (越小越穩定)")

# 過擬合分析
overfitting_issues = results_df[results_df['overfitting'] > 0.1]
if len(overfitting_issues) > 0:
    print(f"⚠️  過擬合問題: {', '.join(overfitting_issues['target'].tolist())}")
else:
    print("✅ 無明顯過擬合問題")

print(f"\n💡 改進建議")
print("-" * 50)

avg_r2 = results_df['holdout_r2'].mean()
if avg_r2 >= 0.7:
    print("🎉 整體表現優秀！建議：")
    print("   • 嘗試集成學習 (Voting/Stacking)")
    print("   • 精細超參數調優")
    print("   • 添加更多領域特徵")
elif avg_r2 >= 0.5:
    print("👍 表現良好，建議：")
    print("   • 嘗試XGBoost/LightGBM")
    print("   • 特徵選擇優化")
    print("   • 增加模型複雜度")
else:
    print("🔧 需要重大改進：")
    print("   • 檢查特徵工程")
    print("   • 嘗試不同算法")
    print("   • 考慮數據質量")

# 具體目標建議
worst_target = results_df.loc[results_df['holdout_r2'].idxmin(), 'target']
print(f"\n針對表現最差的 {worst_target}:")
print("   • 考慮領域特定特徵")
print("   • 檢查數據分布")
print("   • 嘗試非線性模型")

# 儲存結果
results_df.to_csv("final_comprehensive_results.csv", index=False)

print(f"\n💾 詳細結果已儲存至: final_comprehensive_results.csv")

print(f"\n" + "=" * 70)
print("🎯 改進完成總結")
print("=" * 70)
print(f"✅ 使用組合特徵: Morgan指紋 + MACCS指紋 + 分子描述符")
print(f"✅ 改進RandomForest參數，防止過擬合")
print(f"✅ 實施5折交叉驗證確保穩健性")
print(f"✅ 提供完整評估指標與診斷")
print(f"✅ 平均性能提升: {avg_improvement:.1f}%")
print("=" * 70)