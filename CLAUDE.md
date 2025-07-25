# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

用繁體中文回答我

## 專案概述

這是一個聚合物性質預測的機器學習研究專案，實作了兩種不同的特徵化方法：
1. **標準指紋方法** (`01_load_inspect.py`): 使用 RDKit 內建指紋 (Morgan, MACCS, Avalon, RDK)
2. **多層次自定義特徵** (`myself_fingerprint/`): 原子、基團、鏈級三層分子表徵

## 環境設置

### 依賴安裝
```bash
conda create -n rdkit-env python=3.10 -y
conda activate rdkit-env
conda install -c conda-forge rdkit pandas numpy scikit-learn -y
```

### 核心依賴
- `rdkit`: 分子化學資訊學處理
- `pandas`, `numpy`: 資料操作
- `sklearn`: 機器學習模型與預處理

## 資料結構

- **輸入檔案**: `data/train.csv`, `data/test.csv`
- **SMILES**: 聚合物分子結構字符串
- **目標屬性**: Tg (玻璃化轉換溫度), FFV (自由體積分率), Tc (結晶溫度), Density (密度), Rg (回轉半徑)

## 架構說明

### 方法一：標準指紋 (`01_load_inspect.py`)

完整的端到端 ML 流程：
- **特徵**: Morgan (2048位) + MACCS (167位) + Avalon (512位) + RDK (2048位) = 4775維
- **模型**: RandomForest + MultiOutputRegressor (多目標回歸)
- **預處理**: StandardScaler 標準化
- **輸出**: `submission.csv`

### 方法二：多層次特徵 (`myself_fingerprint/`)

#### 三層特徵架構
1. **原子級** (`atom_featurize.py`): 155維
   - 三原子片段 A-B-C 配位數組合
   - 基於鍵連通性拓撲分析
   
2. **基團級** (`group_featurize.py`): 197維
   - 使用 197 個 SMARTS 模式匹配
   - 正規化為每原子出現頻率
   - 依賴 `smarts_list.py` 定義化學基團
   
3. **鏈級** (`chain_featurize.py`): 59維
   - QSPR 描述符 (29項): 分子量、拓撲指數等
   - 形態學指標 (30項): 環結構、側鏈、連通性

#### 特徵整合 (`02_combine_featurize.py`)
- 合併三層特徵 + 頻率特徵 = 412維
- MinMax 標準化至 [0,1]
- 輸出 `features_all_scaled.csv`

## 常用指令

### 標準指紋方法
```bash
# 完整訓練與預測流程
python 01_load_inspect.py
```

### 多層次特徵方法
```bash
# 進入自定義特徵目錄
cd myself_fingerprint/

# 個別計算各層特徵
python atom_featurize.py      # 原子級特徵
python group_featurize.py     # 基團級特徵  
python chain_featurize.py     # 鏈級特徵

# 整合所有特徵
python 02_combine_featurize.py
```

### 特徵驗證
```bash
# 檢查標準指紋維度
python -c "from rdkit.Chem import AllChem, MACCSkeys; print('總維度:', 2048+167+512+2048)"

# 檢查多層次特徵
python -c "import pandas as pd; df=pd.read_csv('myself_fingerprint/features_all_scaled.csv'); print('Shape:', df.shape)"
```

## 程式碼約定

### 分子處理模式
- 統一使用 `Chem.MolFromSmiles()` 解析 SMILES
- 無效分子以全零向量處理
- 所有特徵函數返回固定長度數值列表

### 模組化設計
- 每個特徵化模組可獨立執行
- 包含完整的 `if __name__ == "__main__"` 測試區塊
- 統一的函數介面設計

### 資料路徑
- 資料檔案位於 `data/` 目錄
- 標準指紋方法使用絕對路徑 `/data/`
- 多層次特徵方法使用相對路徑

## 關鍵檔案

- `smarts_list.py`: 197個化學基團 SMARTS 規則定義
- `01_load_inspect.py`: 完整的標準指紋 ML 流程
- `myself_fingerprint/02_combine_featurize.py`: 多層次特徵整合主程式

## 模型評估

- 驗證集分割: 80/20 (random_state=42)
- 評估指標: MSE (均方誤差)
- 多目標回歸處理 5 種聚合物性質