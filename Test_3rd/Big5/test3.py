## 3.작업형2
'''travel insurance 데이터 - 보험가입여부 예측
- travel insurance 데이터를 활용해서 보험가입여부를 예측하라. 
- 데이터 : https://www.kaggle.com/agileteam/traval-insurance-exam 
80점 나왔을 때 30점?!
- travel_insurance_test03.csv
- travel_insurance_train03.csv
'''

import pandas as pd 
pd.set_option("max_rows", 50)
pd.set_option("max_columns", 20)
train = pd.read_csv('Data/travel_insurance_train03.csv')
test = pd.read_csv('Data/travel_insurance_test03.csv')
# print(train.head())

# print(train[['Unnamed: 0', "TravelInsurance"]])
X_train = train.drop(columns = "TravelInsurance")
y_train = train[['Unnamed: 0', "TravelInsurance"]]
# print(X_train.head())

# print(X_train.info())
# cat : Employment Type, GraduateOrNot, FrequentFlyer, EverTravelledAbroad
# num : Age, AnnualIncome, FamilyMembers, ChronicDiseases

X_num = X_train[['Age', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases']]

# print(X_train[['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']])
X_cat = X_train[['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']]

X_cat_dummies = pd.get_dummies(X_cat)
# print(X_cat_dummies)
# for i in X_cat:
#     print(X_cat[i].unique())

for i in X_cat:
    uni = X_cat[i].unique()
    X_cat[i] = X_cat[i].replace(uni, [0,1])
    
# print(X_cat.info())
X_cat_label = X_cat 



X_train_label = pd.concat([X_num, X_cat], axis = 1)
X_train_dummies = pd.concat([X_num, X_cat_dummies], axis = 1)

# 모델과 전처리 라이브러리 불러오기 
from sklearn.model_selection import * 
from sklearn.linear_model import * 
from sklearn.preprocessing import * 
from sklearn.metrics import * 
from sklearn.tree import * 
from sklearn.ensemble import * 
from sklearn.neighbors import * 

def score_vis(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict_proba(X_train)[:,1]
    test_pred = model.predict_proba(X_test)[:,1]
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    train_roc = roc_auc_score(y_train, train_pred)
    test_roc = roc_auc_score(y_test, test_pred)
    return "{:.4f} {:.4f} {:.4f} {:.4f}".format(train_score, test_score, train_roc, test_roc)

def various_model(X_train, X_test, y_train, y_test):
    model1 = LogisticRegression().fit(X_train, y_train)
    print('model1 : ', score_vis(model1, X_train, X_test, y_train, y_test))
    
    for i in range(3, 8):
        model2 = DecisionTreeClassifier(max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model2와", i , score_vis(model2, X_train, X_test, y_train, y_test))
    
    for i in range(3, 8):
        model3 = RandomForestClassifier(max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model3와", i , score_vis(model3, X_train, X_test, y_train, y_test))
    
    for i in range(3, 8):
        model4 = KNeighborsClassifier(i).fit(X_train, y_train)
        print("model4와", i , score_vis(model4, X_train, X_test, y_train, y_test))
        
# 데이터 전처리 
minmax = MinMaxScaler() 
minmax.fit(X_train_label)
scaled_X = minmax.transform(X_train_label)

# print(y_train)
y = y_train['TravelInsurance']
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, stratify = y, test_size= 0.3, random_state = 410)

various_model(X_train, X_test, y_train, y_test)

minmax = MinMaxScaler()
minmax.fit(X_train_dummies)
scaled_X_dum = minmax.transform(X_train_dummies)

X_train, X_test, y_train, y_test = train_test_split(scaled_X_dum, y, stratify = y, test_size= 0.3, random_state = 410)

various_model(X_train, X_test, y_train, y_test)


test = pd.read_csv('Data/travel_insurance_test03.csv')
X_num = test[['Age', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases']]

X_cat = test[['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']]

X_cat_dummies = pd.get_dummies(X_cat)
X_test_dummies = pd.concat([X_num, X_cat_dummies], axis = 1)
scaled_X_dum_test = minmax.transform(X_test_dummies)

model3 = RandomForestClassifier(max_depth = 6, random_state = 410).fit(X_train, y_train)
pred = model3.predict_proba(scaled_X_dum_test)[:, 1]
print(pred)

submission = pd.DataFrame({'ID': test['Unnamed: 0'],
                           'TravelInsurance': pred})
submission.to_csv('submission.csv', index=False)