#!/usr/bin/env python3
"""
檔案整理腳本：移除不重要的檔案，保留核心內容
"""

import os
import shutil

def organize_files():
    """整理專案檔案"""
    
    print("🗂️  開始整理專案檔案...")
    
    # 建立整理後的目錄結構
    dirs_to_create = [
        "archive",           # 歷史版本存檔
        "results",          # 結果檔案
        "deprecated"        # 過時的檔案
    ]
    
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ 建立目錄: {dir_name}/")
    
    # 檔案分類
    file_categories = {
        # 核心檔案 (保留在根目錄)
        "core": [
            "CLAUDE.md",                    # 專案說明文件
            "01_load_inspect.py",           # 原始模型 (基準)
            "06_final_improved_model.py",   # 最終改進版本
            "improvement_summary.md"        # 改進總結
        ],
        
        # 結果檔案 (移動到 results/)
        "results": [
            "fingerprint_comparison_mse.csv",
            "fingerprint_comparison_r2.csv",
            "comprehensive_model_results.csv",
            "final_comprehensive_results.csv",
            "stage1_improvement_results.csv"
        ],
        
        # 歷史版本 (移動到 archive/)
        "archive": [
            "02_improved_model_stage1.py",     # 階段1改進
            "03_simple_improvement.py",        # 簡單改進測試
            "04_comprehensive_improved_model.py", # 完整版本
            "05_quick_comprehensive_test.py"   # 快速測試版
        ],
        
        # 過時檔案 (移動到 deprecated/)
        "deprecated": [
            "fingerprint_vs_descriptors_demo.py",  # 演示用
            "teach.md",                             # 安裝說明
            "organize_files.py"                     # 本身
        ]
    }
    
    # 執行檔案移動
    for category, files in file_categories.items():
        if category == "core":
            continue  # 核心檔案保留在根目錄
            
        print(f"\n📁 處理 {category} 檔案...")
        target_dir = category
        
        for file_name in files:
            if os.path.exists(file_name):
                try:
                    shutil.move(file_name, f"{target_dir}/{file_name}")
                    print(f"  ✅ 移動: {file_name} → {target_dir}/")
                except Exception as e:
                    print(f"  ❌ 錯誤: {file_name} - {e}")
            else:
                print(f"  ⚠️  檔案不存在: {file_name}")
    
    # 處理 myself_fingerprint 目錄
    if os.path.exists("myself_fingerprint"):
        print(f"\n📁 處理 myself_fingerprint/ 目錄...")
        try:
            shutil.move("myself_fingerprint", "archive/myself_fingerprint")
            print(f"  ✅ 移動: myself_fingerprint/ → archive/")
        except Exception as e:
            print(f"  ❌ 錯誤: myself_fingerprint/ - {e}")
    
    # 建立整理後的目錄說明
    create_readme_files()
    
    print(f"\n🎯 檔案整理完成!")
    print_final_structure()

def create_readme_files():
    """建立各目錄的說明檔案"""
    
    readme_contents = {
        "archive/README.md": """# Archive - 歷史版本
        
這個目錄包含開發過程中的各個版本：

- `02_improved_model_stage1.py` - 階段1改進版本
- `03_simple_improvement.py` - 簡單改進測試  
- `04_comprehensive_improved_model.py` - 完整版本
- `05_quick_comprehensive_test.py` - 快速測試版
- `myself_fingerprint/` - 多層次特徵化方法

這些檔案保留作為開發歷史參考。
""",
        
        "results/README.md": """# Results - 結果檔案

這個目錄包含模型訓練的結果：

- `fingerprint_comparison_*.csv` - 指紋比較結果
- `*model_results.csv` - 模型評估結果

這些檔案包含詳細的性能指標和比較分析。
""",
        
        "deprecated/README.md": """# Deprecated - 過時檔案

這個目錄包含不再需要的檔案：

- 演示腳本
- 安裝說明
- 臨時測試檔案

這些檔案可以考慮刪除。
"""
    }
    
    for file_path, content in readme_contents.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"  📝 建立說明: {file_path}")

def print_final_structure():
    """顯示整理後的目錄結構"""
    
    print(f"\n📂 整理後的專案結構:")
    print("=" * 40)
    print("📁 /")
    print("├── 📄 CLAUDE.md                     # 專案說明")
    print("├── 🐍 01_load_inspect.py            # 原始基準模型")  
    print("├── 🚀 06_final_improved_model.py    # 最終改進版本")
    print("├── 📋 improvement_summary.md        # 改進總結")
    print("├── 📊 data/                         # 資料檔案")
    print("│   ├── train.csv")
    print("│   └── test.csv") 
    print("├── 📁 results/                      # 結果檔案")
    print("│   ├── README.md")
    print("│   └── *.csv")
    print("├── 📁 archive/                      # 歷史版本")  
    print("│   ├── README.md")
    print("│   ├── 02_improved_model_stage1.py")
    print("│   ├── 03_simple_improvement.py")
    print("│   ├── 04_comprehensive_improved_model.py")
    print("│   ├── 05_quick_comprehensive_test.py")
    print("│   └── myself_fingerprint/")
    print("└── 📁 deprecated/                   # 過時檔案")
    print("    ├── README.md")
    print("    └── *.py")
    
    print(f"\n💡 建議:")
    print("✅ 保留: 根目錄的核心檔案")
    print("📊 查看: results/ 中的分析結果") 
    print("📚 參考: archive/ 中的開發歷史")
    print("🗑️  清理: deprecated/ 可考慮刪除")

if __name__ == "__main__":
    organize_files()