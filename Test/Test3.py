import pandas as pd 
pd.set_option("max_rows", 50)
pd.set_option("max_columns", 20)

X_train = pd.read_csv("Data/X_train.csv", encoding="euc-kr")
test = pd.read_csv("Data/X_test.csv", encoding="euc-kr")
y_train = pd.read_csv("Data/y_train.csv", encoding="euc-kr")

# print(X_train.head())

# print(X_train.isnull().sum())
# 환불금액이 많은 결측치를 가지고 있음

# print(X_train.groupby('주구매상품')['환불금액'].mean())
# fillna = X_train.groupby('주구매상품')['환불금액'].mean()
fillna_tran = X_train.groupby('주구매상품')['환불금액'].transform('mean')
# print(fillna)
# print(fillna_tran)
X_train['환불금액'] = X_train['환불금액'].mask(X_train['환불금액'].isna(), fillna_tran)
# print(X_train.isnull().sum())

X_train['환불금액'] = X_train['환불금액'].fillna(X_train['환불금액'].mean())
# print(X_train.isnull().sum())

# print(X_train.info())
# num : 총구매액, 최대구매액, 환불금액, 내점일수, 내점당근무건수, 주말방문비율, 구매주기 
# cat : 주구매상품, 주구매지점

X_cat = X_train[['주구매상품', '주구매지점']]
X_num = X_train[['총구매액', '최대구매액', '환불금액', '내점일수', '내점당구매건수', '주말방문비율', '구매주기']]


for i in X_cat:
    uni = X_cat[i].unique()
    if i == '주구매상품':
        X_cat[i] = X_cat[i].replace(uni, range(0,42))
    else:
        X_cat[i] = X_cat[i].replace(uni, range(0,24))
# print(X_cat.unique)
# for i in X_cat:
#     print(X_cat[i].unique())

    
    
# X_train['주구매지점'] = X_train['주구매지점'].astype('category').cat.codes
# X_train['주구매상품'] = X_train['주구매상품'].astype('category').cat.codes

# for i in X_train[['주구매지점', '주구매상품']]:
#     print(X_train[i].unique())
# print(X_train[['주구매지점', '주구매상품']].unique)
# 주구매상품 : 42개, 주구매지점 : 24개 

X_train = pd.concat([X_num, X_cat], axis = 1)
# print(X_train.info())

from sklearn.model_selection import * 
from sklearn.preprocessing import * 
from sklearn.metrics import * 
from sklearn.linear_model import * 
from sklearn.tree import * 
from sklearn.ensemble import * 
from xgboost import *

y = y_train['gender']

def get_score(model, X_train, X_test, y_train, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:,1]
    
    train_roc = roc_auc_score(y_train, train_prob)
    test_roc = roc_auc_score(y_test, test_prob)
    return "{:.4f} {:.4f} {:.4f} {:.4f}".format(train_score, test_score, train_roc, test_roc)

def models(X_train, X_test, y_train, y_test):
    model1 = LogisticRegression().fit(X_train, y_train)
    print("model1 : " , get_score(model1, X_train, X_test, y_train, y_test))
    
    for i in range(3, 8):
        model2 = DecisionTreeClassifier(max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model2와 깊이 : ", i, get_score(model2, X_train, X_test, y_train, y_test))
        
    for i in range(3, 8):
        model3 = RandomForestClassifier(max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model3와 깊이 : ", i, get_score(model3, X_train, X_test, y_train, y_test))
    
    model5 = XGBClassifier().fit(X_train, y_train)
    print('model5', get_score(model5, X_train, X_test, y_train, y_test))

X_train, X_test, y_train, y_test = train_test_split(X_train, y, stratify=y, test_size = 0.3, random_state=410)

models(X_train, X_test, y_train, y_test)

model = RandomForestClassifier(500, max_depth = 5, random_state = 410).fit(X_train, y_train)
test_prob = model.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, test_prob))


# test셋 변환 
fillna_tran = test.groupby('주구매상품')['환불금액'].transform('mean')
# print(fillna)
# print(fillna_tran)
test['환불금액'] = test['환불금액'].mask(test['환불금액'].isna(), fillna_tran)
# print(X_train.isnull().sum())

test['환불금액'] = test['환불금액'].fillna(test['환불금액'].mean())

X_cat = test[['주구매상품', '주구매지점']]
X_num = test[['총구매액', '최대구매액', '환불금액', '내점일수', '내점당구매건수', '주말방문비율', '구매주기']]

for i in X_cat:
    uni = X_cat[i].unique()
    if i == '주구매상품':
        X_cat[i] = X_cat[i].replace(uni, range(0,41))
    else:
        X_cat[i] = X_cat[i].replace(uni, range(0,24))
X_test = pd.concat([X_num, X_cat], axis = 1)

test_prob = model.predict_proba(X_test)[:,1]


sub = pd.DataFrame({'cust_id' : test['cust_id'],
                   'predict' : test_prob})
print(sub.to_csv('sub.csv', encoding='utf-8', index=False))

