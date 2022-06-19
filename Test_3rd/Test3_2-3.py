''' 2-3.타이타닉 데이터
- 각 열의 결측치 비율을 확인 한 후, 결측치의 비율이 가장 높은 변수명을 구하시오
- 데이터 : https://www.kaggle.com/c/2019-1st-ml-month-with-kakr/overview'''

import pandas as pd 
data = pd.read_csv("Data/titanic_train03.csv", encoding="utf-8")
# print(data.head())

# 결측치 비율을 확인하기 위해 각 column마다 결측값이 몇개인지 찾고 각 column마다 값의 개수가 몇개인지 찾으면 끝
print(data.isnull().sum())
for i in data:
    print(i, "변수의 개수는 : ", len(data[i]))
# 각 column의 길이가 모두 같으므로 그냥 결측값의 개수가 높은 변수만 찾으면 됨
# Age
# 정답 (내가 생각한 것 )

# 다른 사람 풀이
import pandas as pd
titanic = pd.read_csv("Data/titanic_train03.csv")
 
# 결측치 비율 
isna = titanic.isna().sum()
result = isna.index[isna.argmax()]
print(result)  # 정답 : Age