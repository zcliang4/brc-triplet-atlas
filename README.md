---
title: BrC-Triplet-Atlas
emoji: 🟤
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: apache-2.0
library_name: streamlit
---

# BrC-Triplet-Atlas
# 棕碳光敏性质预测平台

棕碳（Brown Carbon, BrC）分子光化学性质的机器学习预测工具。

## 功能 | Features

- **E₀ (V vs SHE)** — 单线态基态能量
- **ET (kJ mol⁻¹)** — 三线态能量  
- **Φ_ISC** — 系间窜越量子产率（低/中/高）
- **E′ (V vs SHE)** — 单电子氧化电位
- **Absorb 等级** — 吸光度预测（低/中/高）

## 部署 | Deployment

### Render.com（推荐）

1. Fork 此仓库
2. 在 [render.com](https://render.com) 创建 Web Service
3. 选择 Docker 部署
4. 选择 Free 实例

## 本地运行 | Local Run

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

## 模型说明

- `PS_ML/` — E₀, ET, Φ_ISC 预测模型
- `final_3class_G2_model/` — Absorb 等级分类器
- `oxi_model-added.model` — 氧化电位预测模型

## 团队 | Team

- 梁展聪（多伦多大学，通讯作者）
- 周丽缘（中国科学院城市环境研究所）
- 常宇清、陈泽强（KAUST）

## 许可证

Apache 2.0
