''' 3.작업형2
travel insurance 데이터 - 보험가입여부 예측
- travel insurance 데이터를 활용해서 보험가입여부를 예측하라. 

80점 나왔을 때 30점?!
- travel_insurance_test03.csv
- travel_insurance_train03.csv
'''

import pandas as pd 
pd.set_option("max_rows", 500) # 출력할 max row 지정 
pd.set_option("max_columns", 20) # 출력할 max columns 지정 

train = pd.read_csv("Data/travel_insurance_train03.csv", encoding="utf-8")
test = pd.read_csv("Data/travel_insurance_test03.csv", encoding="utf-8")

X_train = train.drop(columns = "TravelInsurance")
# TravelInsurance은 종속변수임으로 삭제 

# train[['Unnamed: 0', 'TravelInsurance']].to_csv('y_train.csv', index=False)
# y값 따로 저장 
y = pd.read_csv("y_train.csv")
# print(y.head())

X = pd.concat([X_train, test], ignore_index = True)
print(X.isnull().sum())
# 결측치 없음

# print(X.info())
# object type : Employment Type, GraduateOrNot, FrequentFlyer, EverTravelledAbroad
# print(X[['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']].nunique())
# object 변수들은 2개씩 구성

objcols = ['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']

# for i in objcols:
#     print(X[i].unique())
# Label Encoding 
for i in objcols:
    X[i] = X[i].replace(X[i].unique(), [0, 1])
    # 범주형 변수 dummy 변수로 변경
    # 만약 2개가 아니고 3개 이상이면 범주형 변수를 어떻게 처리할까?
print(X.info())


# 사용할 라이브러리 불러오기
from sklearn.preprocessing import * 
from sklearn.model_selection import * 
from sklearn.linear_model import * 
from sklearn.neighbors import * 
from sklearn.tree import * 
from sklearn.ensemble import * 
# from xgboost import XGBClassifier
from sklearn.metrics import * 

# 모델의 성능을 파악하기 위한 함수
def get_scores(model, X_train, X_test, y_train, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    # 모델 점수 
    
    train_proba = model.predict_proba(X_train)[:,1]
    test_proba = model.predict_proba(X_test)[:,1]
    # 모델 예측값(확률)
    
    train_roc = roc_auc_score(y_train, train_proba)
    test_roc = roc_auc_score(y_test, test_proba)
    
    return "{:.4f} {:.4f} {:.4f} {:.4f}".format(train_score, test_score, train_roc, test_roc)

# 다양한 모델을 만들고 성능을 출력하는 함수 
def make_models(X_train, X_test, y_train, y_test, rs = 410):
    model1 = LogisticRegression().fit(X_train, y_train)
    print("model1의 성능 : ", get_scores(model1, X_train, X_test, y_train, y_test))
    
    # overfitting 우려 존재
    model2 = DecisionTreeClassifier(random_state = rs).fit(X_train, y_train)
    print("model2의 성능 : ", get_scores(model2, X_train, X_test, y_train, y_test))
    
    for d in range(3, 8):
        model2 = DecisionTreeClassifier(max_depth = d, random_state=rs).fit(X_train, y_train)
        print("model2의 성능 및 depth 별 값 : ", d, get_scores(model2, X_train, X_test, y_train, y_test))
        
    model3 = RandomForestClassifier(random_state = rs).fit(X_train, y_train)
    print("model3의 성능 : ", get_scores(model3, X_train, X_test, y_train, y_test))
    
    for d in range(3, 8):
        model3 = RandomForestClassifier(500, max_depth=d, random_state = rs).fit(X_train, y_train)
        print("model3의 성능 및 depth 별 값 : ", d, get_scores(model3, X_train, X_test, y_train, y_test))
        
    # model4 = XGBClassifier(eval_metric='logloss', use_label_encoder = False).fit(X_train, y_train)
    # print("model4의 성능 :", get_scores(model4, X_train, X_test, y_train, y_test))
    

    
# x,y 정의 
# final_X = X.drop(columns=["Unnamed: 0"])
final_X = X
X_use = final_X.iloc[:1490, :]
X_sub = final_X.iloc[1490:, :]
y = y["TravelInsurance"]

# preprocessing 
minmax = MinMaxScaler()
X_scaled_use = minmax.fit(X_use).transform(X_use)
X_scaled_sub = minmax.transform(X_sub)

# train, test 분릴 
X_train, X_test, y_train, y_test = train_test_split(X_scaled_use, y, test_size = 0.3, stratify=y, random_state = 410)

make_models(X_train, X_test, y_train, y_test, 410)

# model2의 depth가 3인것 선택 
model2 = DecisionTreeClassifier(max_depth = 3, random_state=410).fit(X_train, y_train)
print("final_model : ", get_scores(model2, X_train, X_test, y_train, y_test))

pred = model2.predict_proba(X_scaled_sub)[:,1]
submission = pd.DataFrame({"ID" : X_sub["Unnamed: 0"],
                           "TravelInsurance" : pred})
print(submission)
submission.to_csv("submission.csv", index = False)

