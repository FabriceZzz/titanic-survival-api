import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib  # 用于保存和加载模型

# 1. 读取数据
df = pd.read_csv('C:/Users/Fabrice/PycharmProjects/PythonProject/Pandas/train.csv')

# 2. 数据清洗（简单处理，填充缺失值）
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. 特征选择与转换
# 我们只选最简单的几个特征来做预测
features = ['Pclass', 'Sex', 'Age', 'Fare']
X = df[features].copy()
y = df['Survived']


X['Sex']=X['Sex'].map({'male':0, 'female':1})

# 4. 训练模型
print("开始训练模型...")
model = LogisticRegression(max_iter=1000) # max_iter=1000 防止不收敛警告
model.fit(X, y)

print("模型训练完成！")

# 5. 保存模型到本地
# 使用 joblib.dump() 保存模型为 'titanic_model.pkl'
joblib.dump(model, 'titanic_model.pkl')
####