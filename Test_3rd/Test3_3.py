''' 3.작업형2
travel insurance 데이터 - 보험가입여부 예측
- travel insurance 데이터를 활용해서 보험가입여부를 예측하라. 

80점 나왔을 때 30점?!
- travel_insurance_test03.csv
- travel_insurance_train03.csv
'''

import pandas as pd 
train = pd.read_csv("Data/travel_insurance_train03.csv", encoding="utf-8")
# print(train.head())
# y값이 TravelInsurance

test = pd.read_csv("Data/travel_insurance_test03.csv", encoding="utf-8")
# print(test.head())

# unnamed 삭제 
train = train.iloc[:, 1:]
# print(train.head())

# for i in train:
#     print(i, "변수의 값 모음 " , train[i].unique())

X_train = train.iloc[:, :8]
y_train = train["TravelInsurance"]

# print(X_train.head())
# num : Age, AnnualIncome, FamilyMembers 
# cat : Employment, GraduateOrNot, FrequentFlyer, EverTravelledAbroad, ChronicDiseases

X_train_num = X_train[['Age', 'AnnualIncome', 'FamilyMembers']]
X_train_cat = X_train[['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad', 'ChronicDiseases']]

X_test_num = test[['Age', 'AnnualIncome', 'FamilyMembers']]
X_test_cat = test[['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad', 'ChronicDiseases']]

from sklearn.preprocessing import * 
minmax = MinMaxScaler() 
minmax.fit(X_train_num)
X_scaled_train = minmax.transform(X_train_num)
X_scaled_test = minmax.transform(X_test_num)
# print(X_scaled_train)

# 범주형 변수 변환 
X_train_cat_dum = pd.get_dummies(X_train_cat)
X_test_cat_dum = pd.get_dummies(X_test_cat)

# print(len(X_train_cat_dum.columns))
# print(len(X_test_cat_dum.columns))
# print(X_train_cat_dum)

# 열 개수 맞춰줌 
X_train_cat_dum , X_test_cat_dum = X_train_cat_dum.align(X_test_cat_dum, join="inner", axis = 1)

# 범주형 변수와 숫자형 변수들을 하나의 데이터 프레임으로 합침
train_final = pd.concat([pd.DataFrame(X_scaled_train), X_train_cat_dum], axis = 1)
test_final = pd.concat([pd.DataFrame(X_scaled_test), X_test_cat_dum], axis = 1)

from sklearn.linear_model import * 
model = LogisticRegression()
model.fit(train_final, y_train)
# print(model.score(train_final, y_train))

prob_predict = model.predict_proba(test_final)
print(prob_predict)

