# Titanic Survival Predictor API

基于 FastAPI 构建的泰坦尼克号生存预测接口。本项目包含从模型训练、API 封装到 Linux 后台部署的完整工程化流程。

## 🛠 技术栈
- **模型**：Scikit-learn (LogisticRegression)
- **后端**：FastAPI + Uvicorn + Pydantic
- **部署**：Linux (Ubuntu/WSL)

## 📁 项目结构
- `train_model.py`：模型训练脚本，负责清洗数据、训练逻辑回归模型并导出 `titanic_model.pkl`。
- `main.py`：FastAPI 主程序，加载模型并提供 `/predict` 预测接口。
- `requirements.txt`：项目依赖包列表。

## 🚀 快速启动
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
