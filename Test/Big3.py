import pandas as pd 
pd.set_option('max_rows',500)    #출력할 max row를 지정
pd.set_option('max_columns',20)  #출력할 max columns를 지정

X_train = pd.read_csv("Data/X_train.csv", encoding="euc-kr")
print(X_train.head())

X_test = pd.read_csv("Data/X_test.csv", encoding="euc-kr")
print(X_test.head())

y_train = pd.read_csv("Data/y_train.csv", encoding="euc-kr")
print(y_train.head())

total = pd.concat([X_train, X_test], axis = 0, ignore_index = False)
print(total.head())
print('데이터 개수 : ', len(total))

print(total.isnull().sum())
# 환불 금액에서 결측치가 많이 발생

print(total.groupby('주구매상품')['환불금액'].mean())
# 주구매상품들의 평균으로 대체 
print(total.groupby('주구매상품')['환불금액'].transform('mean'))
fill_na = total.groupby('주구매상품')['환불금액'].transform('mean')
total['환불금액'] = total['환불금액'].mask(total['환불금액'].isna(), fill_na)
total['환불금액'] = total['환불금액'].fillna(total['환불금액'].mean())

print(total.isnull().sum())

print(total.info())
# cat : 주구매상품, 주구매지점 
# num : 총구매액, 최대구매액, 환불금액, 내점일수, 내점당구매건수, 주말방문비율, 구매주기

X_cat = total[['주구매상품', '주구매지점']]
X_num = total[['총구매액', '최대구매액', '환불금액', '내점일수', '내점당구매건수', '주말방문비율', '구매주기']]

for i in X_cat:
    print(X_cat[i].unique())
print(pd.get_dummies(X_cat))

print(total['주구매지점'].astype('category').cat.codes)
total['주구매지점'] = total['주구매지점'].astype('category').cat.codes
total['주구매상품'] = total['주구매상품'].astype('category').cat.codes


# 모델 
from sklearn.preprocessing import * 
from sklearn.model_selection import * 
from sklearn.linear_model import * 
from sklearn.ensemble import * 
from sklearn.tree import * 
from sklearn.neighbors import * 
from sklearn.metrics import * 

def get_scores(model, X_train, X_test, y_train, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    prob = model.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, prob)
    return "{:.4f} {:.4f} {:.4f}".format(train_score, test_score, roc)

def train_models(X_train, X_test, y_train, y_test):
    model1 = LogisticRegression(max_iter = 5000).fit(X_train, y_train)
    print("model1 : ", get_scores(model1, X_train, X_test, y_train, y_test))
    
    model2 = DecisionTreeClassifier(random_state = 410).fit(X_train, y_train)
    print("model2 : ", get_scores(model2, X_train, X_test, y_train, y_test))
    
    for i in range(3, 12):
        model2 = DecisionTreeClassifier(max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model2이고 depth는 ", i , get_scores(model2, X_train, X_test, y_train, y_test))
        
    model3 = RandomForestClassifier(random_state= 410).fit(X_train, y_train)
    print("model3 : ", get_scores(model3, X_train, X_test, y_train, y_test))
    
    for i in range(3, 8):
        model3 = RandomForestClassifier(500, max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model3이고 depth는  ", i, get_scores(model3, X_train, X_test, y_train, y_test) )
        

# 데이터 분리
X_train = total.drop(columns=['cust_id'])
X_train = total.iloc[:3500, :]
X_sub = total.iloc[3500:, :]
y_train = y_train['gender']

minmax = MinMaxScaler()
minmax.fit(X_train)
X_scaled_train = minmax.transform(X_train)
X_scaled_test = minmax.transform(X_sub)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_train, y_train, test_size = 0.3, stratify = y_train, random_state = 410)

train_models(X_train, X_test, y_train, y_test)

model = RandomForestClassifier(500, max_depth = 7, random_state = 410).fit(X_train, y_train)
prob = model.predict_proba(X_sub)[:, 1]

submission = pd.DataFrame()
submission['cust_id'] = pd.RangeIndex(3500, 3500+len(X_sub))
submission['gender'] = model.predict_proba(X_sub)[:, 1]
print(submission)

