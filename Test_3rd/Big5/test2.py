### 2-1.캘리포니어 집값 데이터
# 결측치를 포함하는 모든 행을 제거한 후, 처음부터 순서대로 70%를 추출하여, 'housing_median_age' 컬럼의 사분위수 Q1의 값을 구하시오
# - 주의사항 정답 제출시, 정수형으로 제출해야 함

import pandas as pd 
data = pd.read_csv("Data/housing03.csv")
# print(data.head())

# print(len(data) * 0.7)
# 70%는 14448
data2 = data[:14448]
# print(len(data2))

q1 = int(data2['housing_median_age'].quantile(0.25))
# print(q1)
# 19 정답 

### 2-2.연도별 나라별 유병률 데이터
'''2000년 데이터에 대해 국가 전체 유병률 평균 값보다 큰 값을 가진 국가 개수는?
(year 변수와 국가들의 결핵 유병률 값을 가진 데이터셋이 주어짐)

    year  국가이름1  국가이름2 ...
    1990  356        200
    1991  400        150
    1992  340        180
...
'''

import pandas as pd 
pd.set_option('max_rows',500)    #출력할 max row를 지정
pd.set_option('max_columns',20)  #출력할 max columns를 지정
data = pd.read_csv("Data/worlddata.csv")
# print(data.head())

data_2000 = data[data['year'] == 2000]
# print(data_2000.head())

data_2000 = data_2000.iloc[:, 1:]
# print(data_2000.mean())

num = 0.0
for i in data_2000:
    num += data_2000[i].values

# print(num / len(data_2000.columns))
mean_2000 = num / len(data_2000.columns)

cnt = 0 
for i in data_2000:
    if (data_2000[i].values > mean_2000):
        cnt += 1
# print(cnt)
# 76 정답

### 2-3.타이타닉 데이터
# - 각 열의 결측치 비율을 확인 한 후, 결측치의 비율이 가장 높은 변수명을 구하시오
# - 데이터 : https://www.kaggle.com/c/2019-1st-ml-month-with-kakr/overview

import pandas as pd 
pd.set_option("max_rows", 50)
pd.set_option("max_columns", 20)

data = pd.read_csv("Data/titanic_train03.csv")
# print(data.head())

print(data.isnull().sum() / len(data))
# Age가 가장 결측치의 비율이 높음