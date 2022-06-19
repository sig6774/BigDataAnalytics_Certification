''' 2-2.연도별 나라별 유병률 데이터
2000년 데이터에 대해 국가 전체 유병률 평균 값보다 큰 값을 가진 국가 개수는?
(year 변수와 국가들의 결핵 유병률 값을 가진 데이터셋이 주어짐)

    year  국가이름1  국가이름2 ...
    1990  356        200
    1991  400        150
    1992  340        180
...
'''

import pandas as pd 
data = pd.read_csv("Data/worlddata.csv", encoding="utf-8")
# print(data.head())

data_2000 = data[data['year'] == 2000]
# print(data_2000)

data_con = data_2000.iloc[:,1:]
print(data_con)

num = 0.0
for i in data_con:
    num += data_con[i].values
print(num)

print(num / len(data_con.columns))
mean_con = num / len(data_con.columns)

cnt = 0
for i in data_con:
    if data_con[i].values >= mean_con:
        cnt += 1
print(cnt)
# 답 : 76 
# 정답 (내가 생각한 풀이 )

# 다른 사람 풀이
import pandas as pd
 
data = pd.read_csv('./bigdata/worlddata.csv') 
data2000 = data[data['year'] == 2000].T
#print(data2000)
data2000 = data2000.iloc[1:, 0] # 첫번째 행이 'year' 이므로 제거해야 합니다!  
#print(data2000)
mean = data2000.mean()
result = data2000[data2000 > mean].shape[0]
print(result)  # 정답 : 76