from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. 实例化 FastAPI 应用
app = FastAPI(title="泰坦尼克号生存预测器API")

# 2. 加载我们刚才训练好的模型 (在启动时加载一次即可)
model=joblib.load('titanic_model.pkl')

# 3. 定义输入数据的格式 (使用 Pydantic)
# 这就像是一个表单，前端传过来的数据必须符合这个格式
class Passenger(BaseModel):
    Pclass:int
    Sex:str # 注意：前端传 'male' 或 'female'
    Age:float
    Fare:float

# 4. 编写预测接口 (POST 请求)
@app.post("/predict")
def predict_survival(passenger: Passenger):
    """把前端传来的 Pydantic 对象转成字典
     意思是前端传过来的 JSON 数据，
     会自动变成一个 Passenger 对象，你直接用 .Pclass 就能拿到属性。
     """
    data=passenger.dict()

    #把字典转为dataframe,因为模型只认dataframe
    df=pd.DataFrame([data])# 为什么加个 [] 就好了？
    #加上中括号后，data 就从一个“字典”变成了一个“包含一个字典的列表”。
    # 这就相当于告诉 Pandas：“这是一行数据，里面包含这些列的值”。Pandas 立刻就懂了，会建出一个只有 1 行 4 列的标准表格。

    # TODO: 把 Sex 列转换成数字 (跟训练时一样: male=0, female=1)
    df['Sex']=df['Sex'].map({'male':0,'female':1})

    # 5. 让模型进行预测
    # TODO: 获取预测结果 (0 或 1) 和 预测概率
    # 提示: model.predict(df)[0] 可以拿到结果
    # 提示: model.predict_proba(df)[0][1] 可以拿到生还的概率

    print("模型 predict 吐出的原始结果:", model.predict(df))
    print("模型 predict_proba 吐出的原始结果:", model.predict_proba(df))

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # 6. 包装成 JSON 返回
    return {
        "survived": int(prediction),
        "probability": round(float(prob), 2),
        "message": "生还" if prediction == 1 else "遇难"
    }