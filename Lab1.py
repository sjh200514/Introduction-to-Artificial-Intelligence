# 导入所需的库  
import pandas as pd  
import matplotlib.pyplot as plt  

# 从CSV文件中读取Titanic数据集  
data = pd.read_csv("titanic.csv", encoding="utf-8")

# 设置支持中文的字体  
plt.rcParams['font.sans-serif'] = ['SimHei'] 
# 重命名列
data = data.rename(columns={'Survived': '结果'})  
data = data.rename(columns={'Sex': '性别'})  
data = data.rename(columns={'Pclass': '舱位'})
data = data.rename(columns={'Age': '年龄'})  
# 将Sex列中的'female'和'male'替换为'女性'和'男性'  
data['性别'] = data['性别'].replace({'female': '女性', 'male': '男性'}) 
#将Survived列中的0和1替换成'死亡'和'幸存'
data['结果'] = data['结果'].replace({0:'死亡',1:'幸存'})

#计算乘客性别分布
gender_distribution = data['性别'].value_counts()  
print(gender_distribution)

# 可视化性别分布
gender_distribution.plot(kind='bar')  
plt.title("乘客的性别分布")  
plt.xlabel("性别")  
plt.ylabel("乘客数量")  
plt.xticks(rotation=0)
plt.show()

#计算乘客年龄分布
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]  
labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']  
data['年龄区间'] = pd.cut(data['年龄'], bins=bins, labels=labels, right=False)  
age_distribution = data['年龄区间'].value_counts()  
print(age_distribution)  

# 可视化年龄分布  
age_distribution.plot(kind='bar')  
plt.title('乘客的年龄分布')  
plt.xlabel('年龄区间')  
plt.ylabel('乘客数量')  
plt.xticks(rotation=0)  # 旋转x轴标签，以便更好地显示  
plt.show()

# 分析性别对生存概率的影响

# 计算每个性别的生存人数  
survival_by_sex = data.groupby(['性别', '结果']).size().unstack(fill_value=0)  

# 计算每个性别的生存率  
survival_rate_by_sex = survival_by_sex.div(survival_by_sex.sum(axis=1), axis=0) * 100  
# 打印结果  
print(survival_rate_by_sex)  

# 可视化结果  
survival_rate_by_sex.plot(kind='bar', stacked=False, figsize=(10, 6))  
plt.title('性别对生还概率的影响')  
plt.xlabel('性别')  
plt.ylabel('幸存/死亡比例 (%)')  
plt.xticks(rotation=0)  
plt.legend(title='是否存活', labels=['幸存', '死亡'])  
plt.show()

# 分析舱位等级对生存概率的影响 
data['舱位'] = data['舱位'].replace({1: '一等舱', 2: '二等舱', 3:'三等舱'})
ordered_pclass = pd.Categorical(data['舱位'], categories=['一等舱', '二等舱', '三等舱'], ordered=True)  
data['舱位'] = ordered_pclass 

survival_by_pclass = data.groupby(['舱位', '结果']).size().unstack(fill_value=0) 

survival_rate_by_pclass = survival_by_pclass.div(survival_by_pclass.sum(axis=1), axis=0) * 100  

print(survival_rate_by_pclass)  

survival_rate_by_pclass.plot(kind='bar', stacked = False, figsize=(10, 6))  
plt.title('船舱对生还概率的影响')  
plt.xlabel('舱位等级(Pclass)')  
plt.ylabel('幸存/死亡比例 (%)')  
plt.xticks(rotation=0)  
plt.legend(title='是否存活', labels=['幸存', '死亡'])  
plt.show()

#分析年龄对生存概率的影响
bins = [0, 12, 18, 60, 100]    
labels = ['儿童(0-12)', '青少年(12-18)', '青年人(18-60)', '老年人(60-100)']  
data['年龄区间'] = pd.cut(data['年龄'], bins=bins, labels=labels, right=False)  
  
survival_by_age = data.groupby(['年龄区间', '结果']).size().unstack(fill_value=0)  
  
survival_rate_by_age = survival_by_age.div(survival_by_age.sum(axis=1), axis=0) * 100  
  
survival_rate_by_age.plot(kind='bar', stacked=False, figsize=(10, 6))  
print(survival_rate_by_age)

plt.title('年龄对生还概率的影响')  
plt.xlabel('年龄区间')  
plt.ylabel('生存率 (%)')  
plt.xticks(rotation=0)  
plt.legend(title='是否存活', labels=['幸存', '死亡'])  
plt.show()