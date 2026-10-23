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

## 训练并生成模型（如果本地没有 pkl 文件）：

bash
python train_model.py
启动 API 服务（开发模式）：

bash
uvicorn main:app --reload
访问交互式 API 文档：
👉 http://127.0.0.1:8000/docs

## ⚙️ 生产环境部署 (Linux)
在 Linux 服务器上，使用 nohup 命令使服务脱离终端常驻后台运行：

bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
接口地址：http://<服务器IP>:8000/predict

日志监控：tail -f api.log

## 📝 接口测试示例
使用 curl 或 Python requests 发送 POST 请求：

json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 28,
  "Fare": 100.5
}
返回：

json
{
  "survived": 1,
  "probability": 0.93,
  "message": "生还"
}
