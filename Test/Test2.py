## 2.작업형 1 예시문제
# mtcars 데이터셋(mtcars.csv)의 qsec 컬럼을 최소 최대 척도(Min-Max Scale)로 변환한 후 
# 0.5보다 큰 값을 가지는 레코드 수를 구하시오.

import pandas as pd 
pd.set_option("max_rows", 50)
pd.set_option("max_columns", 20)
data = pd.read_csv("Data/mtcars.csv", encoding="utf-8")
print(data.head())

def minmaxscaled(series):
    num = (series - series.min()) / (series.max() - series.min())
    return num
d = minmaxscaled(data['qsec'])
print(d)

cnt = 0 
for i in d:
    if i >= 0.5:
        cnt += 1
print(cnt)
# 9 


#### 2-1. 보스턴 데이터
# 보스턴 데이터 범죄율 컬럼 top10 중 
# 10번째 범죄율 값으로 1~10위의 범죄율 값을 변경 후 AGE 변수 80이상의 범죄율 평균을 산출하라.

import pandas as pd 
pd.set_option("max_rows", 20)
pd.set_option("max_columns", 30)

data = pd.read_csv("Data/boston_housing.csv", encoding="utf-8")
print(data.head())

print(data.sort_values(by = "crim", ascending=False))
sort_val = data.sort_values(by="crim", ascending=False)
print(sort_val['crim'][:10])
# 10번째 값 : 25.9406

def trans(s):
    if s >= 25.9406:
        return 25.9406
    
    else:
        return s
    
sort_val['new_crim'] = sort_val['crim'].apply(trans)
print(sort_val)

print(sort_val[sort_val['age'] >= 80]['new_crim'].mean())
# 값 : 5.759386625

#### 2-2. 하우징 데이터
# 주어진 데이터 첫번째 행 부터 순서대로 80%까지의 데이터를 추출 후 'total_bedrooms' 변수의 결측값(NA)을 'total_bedrooms' 변수의 
# 중앙값으로 대체하고 대체 전의 'total_bedrooms' 변수 표준편차값과 대체 후의 'total_bedrooms' 변수 표준편차 값 산출
import pandas as pd 
pd.set_option("max_rows", 20)
pd.set_option("max_columns", 20)

data = pd.read_csv("Data/housing.csv", encoding="utf-8")
print(data.head())
print(len(data)*0.8)
data2 = data[:16512]
print(len(data2))

NotTransStd = data2['total_bedrooms'].std()
print(NotTransStd)

data2['total_bedrooms'] = data2['total_bedrooms'].fillna(data2['total_bedrooms'].median())
TransStd = data2['total_bedrooms'].std()
print(TransStd)

print(abs(NotTransStd - TransStd))
# 1.9751472916

#### 2-3. 하우징 데이터 
# 데이터의 특정컬럼(latitude)의 이상치(이상치 기준 : 평균 + (표준편차 * 1.5) )를 찾아 이상치들의 합 산출 
import pandas as pd 
pd.set_option("max_rows", 20)
pd.set_option("max_columns", 20)

data = pd.read_csv("Data/housing.csv", encoding="utf-8")
print(data.head())

lati_mean = data['latitude'].mean()
lati_std = data['latitude'].std()

plus_outlier = lati_mean + (lati_std * 1.5)
minus_outlier = lati_mean - (lati_std * 1.5)

num = 0
for i in data['latitude']:
    if i >= plus_outlier:
        num += i 
    elif i <= minus_outlier:
        num += i 
print(num)
# 답 : 45815.7500000000..
