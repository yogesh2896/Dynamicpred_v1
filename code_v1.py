def xgb(c1,c2):

# XGBoost Algorithm
  import pandas as pd
  import streamlit as st
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  import matplotlib.pyplot as plt
  import seaborn as sns
  import matplotlib.pyplot as plt
  import plotly.express as px
  # sdate=st.date_input(label='select start d
  # edate=st.date_input(label='select end date')ate')
  import time
  import streamlit as st

  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(5)
  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(5)
  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(5)





  df=pd.read_csv(url)
  df=df[['date','Sales_Value']]
  df['date'] = pd.to_datetime(df['date'],format='mixed')
  df['date'] = pd.to_datetime(df[c1]).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df.columns=['date','Sales_Value']
  q1=np.percentile(df['Sales_Value'],25)
  q3=np.percentile(df['Sales_Value'],75)
  iqr=q3-q1
  ll=q1-1.5*iqr

  ul=q3+1.5*iqr
  df=df[(df['Sales_Value']>ll) & (df['Sales_Value']<ul)]








  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  l=round(len(df)/3)
# xtrain=x[:-l]
# xtest=x[-l:]
# ytrain=y[:-l]
# ytest=y[-l:]
  params = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
  }
  model=XGBRegressor()
  randomized_search = RandomizedSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
  randomized_search.fit(xtrain, ytrain)
  best_params = randomized_search.best_params_
  model=XGBRegressor(n_estimators=best_params['n_estimators'],min_child_weight=best_params['min_child_weight'],max_depth=best_params['max_depth'],
                   learning_rate=best_params['learning_rate'],gamma=best_params['gamma'],colsample_bytree=best_params['colsample_bytree'])
  model=model.fit(xtrain,ytrain)
  ypred=model.predict(xtest)
  comp_df=pd.DataFrame(ypred,ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns=['predicted','actual']

  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write(r"$\color{white}\textsf{\small Accuracy Score}$",score,"%")
  print('Below prediction based on XGBoost Regressor Algorithm')
  # plt.plot(comp_df['predicted'], label='Predicted')
  # plt.plot(comp_df['actual'], label='Actual')
  # plt.xlabel('Value')
  # plt.ylabel('Frequency')
  # _ = plt.legend()
  # plt.show()
  # comp_df.reset_index(inplace=True)
  # comp_df.set_index('index',inplace=True)
  # # print(comp_df)
  # import plotly.express as px
  # fig=px.line(comp_df,x=comp_df['index'],y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  # # st.write(fig)
  # fig.show()
  # print(comp_df)
  # sns.lineplot(x=comp_df.index,y=comp_df['predicted'])
  # sns.lineplot(x=comp_df.index,y=comp_df['actual'])
  # plt.show()

  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  fig.update_layout(width=1130,height=500)
  st.write(fig)
  print(r2_score(comp_df['actual'],comp_df['predicted'])*100)
  # fig.show()

  # sdate=pd.to_datetime(sdate)
  # edate=pd.to_datetime(edate)
  # sdate='02/02/2024'
  # edate='02/04/2024'
  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  pfig.update_layout(width=1130,height=500)
  st.write(pfig)
  st.dataframe(newdf)






















































# ************************************************************************************************************************************************************
def sarimax(c1,c2):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  from pmdarima import auto_arima
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  import warnings
  from sklearn import metrics
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  import streamlit as st
  import pandas as pd
  from statsmodels.tsa.seasonal import seasonal_decompose
  import plotly.graph_objs as go
  from statsmodels.tsa.seasonal import seasonal_decompose
  import plotly.express as px
  import time
  import streamlit as st

  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(20)

  df=pd.read_csv(url)
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date'],format='mixed')
  df.columns=['date','value']
  #monthwise plot
  mdf=df.groupby(df['date'].dt.month).sum('value')
  
  fig1=px.line(mdf,x=mdf.index,y='value',color_discrete_sequence=px.colors.qualitative.Plotly,title='Month wise seasonality')
  fig1.update_layout(width=1130,height=500)
  st.write(fig1)
  # weekwiseplot
  wdf=df.groupby(df['date'].dt.isocalendar().week).sum('value')
  
  fig1=px.line(wdf,x=wdf.index,y='value',color_discrete_sequence=px.colors.qualitative.Plotly,title='Week wise seasonality')
  fig1.update_layout(width=1130,height=500)
  st.write(fig1)
 


  df['date']=pd.to_datetime(df['date'])
  df['date'] = df['date'].apply(lambda x: x.replace(day=1))
  df=df.groupby('date').sum('value')

 



  import plotly.express as px
  res=seasonal_decompose(df['value'])
  seasonal = res.seasonal
  trend = res.trend
  resid = res.resid
  df_decomposed = pd.DataFrame({
    "seasonal": seasonal,
    "trend": trend,
    "resid": resid,
})

  fig = px.line(df_decomposed, y="trend", title="Trend Decomposition")
  fig.update_layout(width=1130,height=500)
  st.write(fig)
  fig = px.line(df_decomposed, y="seasonal", title="Seasonal Decomposition")
  fig.update_layout(width=1130,height=500)
  st.write(fig)






  q1=np.percentile(df['value'],25)
  q3=np.percentile(df['value'],75)
  iqr=q3-q1
  ll=q1-1.5*iqr
  # ll=19000
  ul=q3+1.5*iqr
  df=df[(df['value']>ll) & (df['value']<ul)]

  # df['date']=pd.to_datetime(df['date'])
  # df['date'] = df['date'].apply(lambda x: x.replace(day=1))
  # df=df.groupby('date').sum('value')

  autoari=auto_arima(df,seasonal=True,maxiter=300,suppress_warnongs=True)
  l=round(len(df)/3)
  train=df[:-l]
  test=df[-l:]
  acc_score=[]
  params=[]
  for i in range(0,4):
    p=autoari.order[0]+i
    d=autoari.order[1]+i
    q=autoari.order[2]+i
    model=SARIMAX(train,order=(p,d,q),seasonal_order=(p,d,q,12),trend=None)
    model=model.fit()

    #accuracy checking
    acc_model=model.get_forecast(len(test))
    # print(type(acc_model.predicted_mean))
    # print(type(test['value']))
    a=metrics.mean_absolute_error(test['value'],acc_model.predicted_mean)
    params.append(i)
    acc_score.append(a)
  acc_df=pd.DataFrame(params,acc_score)
  # acc_df.columns=['i','a']
  acc_df=acc_df.reset_index()
  acc_df.columns=['value','params']
  acc_df=acc_df.sort_values(by='params')
  best_param=acc_df['params'][1]
  # print(best_param)
  # print(min(acc_df['value']))

  p=autoari.order[0]+best_param
  d=autoari.order[1]+best_param
  q=autoari.order[2]+best_param
  model=SARIMAX(train,order=(p,d,q),seasonal_order=(p,d,q,12),trend=None)
  model=model.fit()

  trained_model=model.get_forecast(len(test)+12)

  predictions=trained_model.predicted_mean

  predictions=pd.DataFrame(predictions)



  predictions=predictions.reset_index()

  df=df.reset_index()
  predictions.columns=df.columns
  df=df.merge(predictions,how='outer',on='date')
  df.columns=['date','Actual','Predicted']
  print("Below predictions based on SARIMAX Algorithm")

  # sns.lineplot(x=df['date'],y=df['value_x'],label='Actual')
  # fig=sns.lineplot(x=df['date'],y=df['value_y'],label='Predicted')
  # plt.legend()
  # plt.show()
  # st.write(fig)
  # print(df)

  import plotly.express as px
  fig=px.line(df,x=df['date'],y=['Actual','Predicted'],color_discrete_sequence=px.colors.qualitative.Plotly)
  fig.update_layout(width=1130,height=500)
  st.write(fig)

# ***********************************************************************************************************************************
import streamlit as st
def linearregression(c1,c2):
  import pandas as pd
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  from lightgbm import LGBMRegressor
  import matplotlib.pyplot as plt
  import plotly.express as px
# !pip install streamlit
# Read data
  import time
  import streamlit as st

  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(10)

  df=pd.read_csv(url,parse_dates=['date'])
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date'],format='mixed')
  df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  parameters = {'fit_intercept':[True,False],'copy_X':[True, False]}
  grid = RandomizedSearchCV(LinearRegression(), parameters, cv=5, n_jobs=-1, verbose=2)
  grid.fit(xtrain, ytrain)
  print(grid.best_params_)

  best_params=grid.best_params_
  best_params
  model=LinearRegression(fit_intercept=best_params['fit_intercept'],copy_X=best_params['copy_X'])
  model=model.fit(xtrain,ytrain)
  ypred=model.predict(xtest)
  comp_df=pd.DataFrame(ypred,ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns=['predicted','actual']



  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  fig.update_layout(width=1130,height=500)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write(r"$\color{white}\textsf{\small Accuracy Score}$",score,"%")

  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  pfig.update_layout(width=1130,height=500)
  st.write(pfig)
  st.dataframe(newdf)
# *******************************************************************************************
def decisiontree(c1,c2):
  import pandas as pd
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  from lightgbm import LGBMRegressor
  import matplotlib.pyplot as plt
  import plotly.express as px
  from sklearn.tree import DecisionTreeRegressor
# !pip install streamlit
# Read data
  import time
  import streamlit as st

  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(10)
  df=pd.read_csv(url,parse_dates=['date'])
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date'],format='mixed')
  df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  parameters = {'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10]}
  grid = RandomizedSearchCV(DecisionTreeRegressor(), parameters, cv=5, n_jobs=-1, verbose=2)
  grid.fit(xtrain, ytrain)
  print(grid.best_params_)
  best_params = grid.best_params_
  model = DecisionTreeRegressor(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
  model.fit(xtrain, ytrain)
  ypred = model.predict(xtest)
  comp_df = pd.DataFrame(ypred, ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns = ['predicted', 'actual']



  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  fig.update_layout(width=1130,height=500)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write(r"$\color{white}\textsf{\small Accuracy Score}$",score,"%")

  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  pfig.update_layout(width=1130,height=500)
  st.write(pfig)
  st.dataframe(newdf)

# ********************************************************************************************

def randomforestregressor(c1,c2):
  import pandas as pd
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  from lightgbm import LGBMRegressor
  import matplotlib.pyplot as plt
  import plotly.express as px
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.ensemble import RandomForestRegressor
# !pip install streamlit
# Read data
  import time
  import streamlit as st

  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(10)
  df=pd.read_csv(url,parse_dates=['date'])
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date'],format='mixed')
  df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  parameters = {'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10]}
  grid = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=5, n_jobs=-1, verbose=2)
  grid.fit(xtrain, ytrain)
  print(grid.best_params_)
  best_params = grid.best_params_
  model = RandomForestRegressor(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
  model.fit(xtrain, ytrain)
  ypred = model.predict(xtest)
  comp_df = pd.DataFrame(ypred, ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns = ['predicted', 'actual']



  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  fig.update_layout(width=1130,height=500)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write(r"$\color{white}\textsf{\small Accuracy Score}$",score,"%")

  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  pfig.update_layout(width=1130,height=500)
  st.write(pfig)
  st.dataframe(newdf)

# *********************************************************************
# prompt: lightgbm hyperametes

def lightgbmregressor(c1,c2):
  import pandas as pd
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  from lightgbm import LGBMRegressor
  import matplotlib.pyplot as plt
  import plotly.express as px
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.ensemble import RandomForestRegressor



  # !pip install streamlit
  # Read data
  import time
  import streamlit as st

  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(10)
  df=pd.read_csv(url,parse_dates=['date'])
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date'],format='mixed')
  df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  parameters = {'num_leaves': [31, 50, 100],
              'max_depth': [5, 8, 10],
              'learning_rate': [0.01, 0.1, 0.2]}
  grid = RandomizedSearchCV(LGBMRegressor(), parameters, cv=5, n_jobs=-1, verbose=2)
  grid.fit(xtrain, ytrain)
  print(grid.best_params_)
  best_params = grid.best_params_
  model = LGBMRegressor(num_leaves=best_params['num_leaves'], max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'])
  model.fit(xtrain, ytrain)
  ypred = model.predict(xtest)
  comp_df = pd.DataFrame(ypred, ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns = ['predicted', 'actual']
  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  pfig.update_layout(width=1130,height=500)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write(r"$\color{white}\textsf{\small Accuracy Score}$",score,"%")
  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  pfig.update_layout(width=1130,height=500)
  st.write(pfig)
  st.dataframe(newdf)
  # ************************************************************************************

def svmregressor(c1,c2):
  import pandas as pd
  import streamlit as st
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.svm import SVR
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  import matplotlib.pyplot as plt
  import seaborn as sns
  import plotly.express as px
  from pmdarima import auto_arima
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  from statsmodels.tsa.seasonal import seasonal_decompose
  import warnings
  from sklearn import metrics
  import streamlit as st
  from sklearn.linear_model import LinearRegression
  from lightgbm import LGBMRegressor
  from sklearn.tree import DecisionTreeRegressor
  import plotly.graph_objs as go
  from sklearn.ensemble import RandomForestRegressor
  from PIL import Image
  import requests
  from io import BytesIO
  import json
  import requests  # pip install requests
  import streamlit as st  # pip install streamlit







# !pip install streamlit
# Read data
  import time
  import streamlit as st

  with st.spinner(r"$\color{white}\textsf{\small Machine Learning Algorithm running...}$"):
    time.sleep(15)
  df=pd.read_csv(url,parse_dates=['date'])
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date'],format='mixed')
  df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'C': [10, 100, 1000, 2000],
              'gamma': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
  grid = RandomizedSearchCV(SVR(), parameters, cv=5, n_jobs=-1, verbose=2)
  grid.fit(xtrain, ytrain)
  print(grid.best_params_)
  best_params = grid.best_params_
  model = SVR(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
  model.fit(xtrain, ytrain)
  ypred = model.predict(xtest)
  comp_df = pd.DataFrame(ypred, ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns = ['predicted', 'actual']



  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  fig.update_layout(width=1130,height=500)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write(r"$\color{white}\textsf{\small Accuracy Score}$",score,"%")

  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  pfig.update_layout(width=1130,height=500)
  st.write(pfig)
  st.dataframe(newdf)



# ************************************************************************************************************************
# Front end Code
import streamlit as st
st.set_page_config(layout="wide")

imurl="https://enoahisolution.com/wp-content/themes/enoah-new/images/newimages/enoah-invert-logo.png"
# st.image(im)
from PIL import Image
import requests
from io import BytesIO




# row1_col1,row1_col2=st.columns(2)
# row2_col1,row2_col2=st.columns(2)

col1, col2, col3 = st.columns([4,5,5])

response = requests.get(imurl)
img = Image.open(BytesIO(response.content))
new_image = img.resize((300,300))
with col1:
  st.image(img,width=200)















import streamlit as st
import json

import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie

# GitHub: https://github.com/andfanilo/streamlit-lottie
# Lottie Files: https://lottiefiles.com/

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# lottie_coding = load_lottiefile("https://lottie.host/42865cd6-f214-477d-8983-89fe38a0ee89/T8eJueRad9.json")  # replace link to local lottie file
lottie_hello = load_lottieurl("https://lottie.host/214ec299-9801-47e4-913b-45c904582263/AOIyqhKSoU.json")










col1, col2, col3 = st.columns([2,4,2])
with col2:
  st_lottie(
    lottie_hello,
    speed=2,
    reverse=False,
    loop=True,
    quality="medium", # medium ; high
    # renderer="svg", # canvas
    height=300,
    width=300,
    key="ML",
)















# st.image('https://static.scientificamerican.com/sciam/cache/file/D78BCB5B-A9C5-4049-A91EE4149D222A85_source.jpg?w=1350')

new_title = '<p style="font-family:Fantasy; color:Litegreen;background-color: White;opacity: 0.9; font-size: 35px;">Forecasting Model by Machine Learning</p>'
st.markdown(new_title, unsafe_allow_html=True)




url=st.file_uploader(label=r"$\color{white}\textsf{\Large Upload your csv file}$",type='csv')
if url is not None:
  st.success(r"$\color{white}\textsf{\large File Uploaded Successfully!}$")
else:
  st.warning(r"$\color{white}\textsf{\large Please upload your csv file}$")











st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://static.scientificamerican.com/sciam/cache/file/D78BCB5B-A9C5-4049-A91EE4149D222A85_source.jpg?w=1350") center;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)








# original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Streamlit CSS Styling✨ </h1>'
# st.markdown(original_title, unsafe_allow_html=True)


# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://static.vecteezy.com/system/resources/previews/026/827/746/non_2x/artificial-intelligence-technology-concept-ai-vector.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)



input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  // This changes the text color inside the input box
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)




# Functions
if st.button(r"$\color{green}\textsf{\small Default Prediction}$"):
  sarimax('date','Sales_Value')



sdate=st.date_input(label=r"$\color{white}\textsf{\small select start date}$")
edate=st.date_input(label=r"$\color{white}\textsf{\small select end date}$")

if sdate>edate:
  st.warning(r"$\color{white}\textsf{\small Please select proper date}$",icon="⚠️")

if st.button(r"$\color{green}\textsf{\small Prediction for selected Date Range with Best Algorithm}$"):
  xgb('date','Sales_Value')
a=st.selectbox(r"$\color{white}\textsf{\small For Other ML Algorithms}$",('Linear Regression','XGBoost Regression','Decision Tree Regression','Random Forest Regression','LightGBM Regression','Support Vector Machine Regression'))

if st.button(r"$\color{green}\textsf{\small Run Algorithm}$"):
  if a=='Linear Regression':
    linearregression('date','Sales_Value')
  if a=='XGBoost Regression':
    xgb('date','Sales_Value')
  if a=='Decision Tree Regression':
    decisiontree('date','Sales_Value')
  if a=='Random Forest Regression':
    randomforestregressor('date','Sales_Value')
  if a=='LightGBM Regression':
    lightgbmregressor('date','Sales_Value')
  if a=='Support Vector Machine Regression':
    svmregressor('date','Sales_Value')


# **********************************************************
# from streamlit_lottie import st_lottie

