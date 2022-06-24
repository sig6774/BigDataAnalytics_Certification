import pandas as pd 
pd.set_option("max_rows", 50)
pd.set_option("max_columns", 20)

data = pd.read_csv("Data/house_price.csv", encoding="utf-8")
# print(data.head())

# print(data.corr())
# 종속 변수인 house_value랑 높은 상관관계가 있는게 income 

# income 지우고 돌려보고 지우지 않고 돌려봐서 비교 
X = data.iloc[:, :5]
print(X)
y = data['house_value']



data2 = data[['housing_age', 'bedrooms', 'households', 'rooms']]
X_income = data2.iloc[:, :4]
y_income = data['house_value']

from sklearn.model_selection import * 
from sklearn.metrics import * 
from sklearn.preprocessing import * 
from sklearn.tree import * 
from sklearn.linear_model import * 
from sklearn.ensemble import * 
from xgboost import * 

minmax = MinMaxScaler() 
minmax.fit(X)
X_scaled = minmax.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 410, test_size = 0.3)

import numpy as np 

def get_scores(model, X_train, X_test, y_train, y_test):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    MSE_train = mean_squared_error(y_train, pred_train)
    MSE_test = mean_squared_error(y_test, pred_test)
    
    RMSE_train = np.sqrt(MSE_train)
    RMSE_test = np.sqrt(MSE_test)
    
    return "{:.4f} {:.4f}".format(RMSE_train, RMSE_test)

def models(X_train, X_test, y_train, y_test):
    model1 = LinearRegression().fit(X_train, y_train)
    print("model1 : ", get_scores(model1, X_train, X_test, y_train, y_test))
    
    for i in range(3,8):
        model2 = DecisionTreeRegressor(max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model2 와 i는 ", i,  get_scores(model2, X_train, X_test, y_train, y_test))
        
    for i in range(3, 8):
        model3 = RandomForestRegressor(max_depth = i, random_state = 410).fit(X_train, y_train)
        print("model3 와 i는", i , get_scores(model3, X_train, X_test, y_train, y_test))
    
    model4 = XGBRegressor().fit(X_train, y_train)
    print("model4 : ", get_scores(model4, X_train, X_test, y_train, y_test))

models(X_train, X_test, y_train, y_test)



# income 삭제하고 결과 
# X_train, X_test, y_train, y_test = train_test_split(X_income, y_income, random_state = 410, test_size = 0.3)

# models(X_train, X_test, y_train, y_test)
# 결과가 너무 안좋기 떄문에 income 삭제하는 건 좋은 생각이 아님

    
# model3와 깊이 7로 설정
model = RandomForestRegressor( max_depth = 7, random_state = 410).fit(X_train, y_train)
pred = model.predict(X_test)

sub = pd.DataFrame({'id' : range(0, len(pred)),
                   'predict' : pred})

sub.to_csv("sub_regression.csv", encoding="utf-8", index=False)
