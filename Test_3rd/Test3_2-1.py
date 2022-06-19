### 2-1.캘리포니어 집값 데이터
### 결측치를 포함하는 모든 행을 제거한 후, 처음부터 순서대로 70%를 추출하여, 'housing_median_age' 컬럼의 사분위수 Q1의 값을 구하시오
    ### - 주의사항 정답 제출시, 정수형으로 제출해야 함
    
import pandas as pd 
data = pd.read_csv("Data/housing03.csv", encoding = "utf-8")
# print(data.head())

# print(data.isnull().sum())
# print(len(data))

data2 = data.dropna()
# print(data2.isnull().sum())
print(len(data2))
# 결측치 제거 완료 

print(len(data2)*0.7)
# 14303 

data3 = data2[:14303]
print(len(data3))

print(int(data3["housing_median_age"].quantile(0.25)))


