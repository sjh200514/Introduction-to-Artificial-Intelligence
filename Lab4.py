import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#获取训练集和测试集的信息
print("训练集整体信息:")
print(train.info())
print("\n测试集整体信息:")
print(test.info())
print("训练集描述性统计信息:")
print(train.describe())
print("\n测试集描述性统计信息:")
print(test.describe())
print("训练集缺失值统计信息:")
print(train.isnull().sum())
print("\n测试集缺失值统计信息:")
print(test.isnull().sum())

#缺失值处理
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].mode()[0], inplace=True)

#将分类数据转换为数值数据
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


traindata = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
trainlabel = train['Survived']
testdata = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

#训练逻辑回归模型
clf = LogisticRegression(max_iter=1000)
clf.fit(traindata, trainlabel)

#预测存活情况
predictions = clf.predict(testdata.drop('PassengerId', axis=1))
    
#将结果导出为CSV文件
result = testdata[['PassengerId']].copy()
result['Survived'] = predictions
result.to_csv('survival_predictions.csv', index=False)
