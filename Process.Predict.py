
# coding: utf-8

# # Predict Water Amount Needed

# ## Get and Prepare Weather Data from DataBase

# In[1]:

import pymysql.cursors
import pandas as pd
import numpy as np

pd.options.display.max_rows = 200
#pd.set_option('display.float_format', lambda x: '%.20f' % x) #Display as Float
pd.set_option('display.float_format', lambda x: '{:,}'.format(x)) #Display as Scientific


connection = pymysql.connect(host = "mikmak.cc", user="sensor", passwd="Gaffe2017", db="weatherDW")
query = ('SELECT * FROM log_v_last24Hours WHERE S_Text <> "None"')

with connection.cursor() as cursor:
    cursor.execute(query)
connection.commit()
e_Log = cursor.fetchall()
connection.close()

e_Log = (np.array(e_Log))


# In[2]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

t_Log = e_Log

t_Log = pd.DataFrame(data = t_Log[1:,:],
                     index = t_Log[1:,2],
                     columns =["ID","Value","TimeStamp"] )

t_Log = t_Log[t_Log != '-'] #DataSet contains some missing values, remove them

t_Log = t_Log.pivot(index='TimeStamp', columns='ID', values='Value')
t_Log = t_Log.sort_index(ascending = True) #It should be ascending, for rolling calculation later
t_Log = t_Log.dropna(how = "any")
t_Log = t_Log.astype(float)

t_Log.tail()


# ## Transform Data into Input Vector

# ### Order Data and Calculate Means and Sums

# In[3]:

X = pd.DataFrame()

lastRow = len(t_Log.axes[0]) - 1

X = X.assign(tre200b0=t_Log.Temperature[[lastRow]])
X = X.assign(ure200b0=t_Log.Humidity[[lastRow]])
X = X.assign(rre200b0=t_Log.Rain[[lastRow]])
X = X.assign(sre000b0=t_Log.Sunshine[[lastRow]])
X = X.assign(fu3010b0=t_Log.Wind[[lastRow]])
X = X.assign(prestab0=t_Log.Pressure[[lastRow]])
#"tre200b0","ure200b0","rre150b0","sre000b0","fu3010b0","prestab0"

t_LogRolling = t_Log.rolling(len(t_Log.axes[0]))

X = X.assign(tre200b0_mean=t_LogRolling.Temperature.mean())
X = X.assign(ure200b0_mean=t_LogRolling.Humidity.mean())
X = X.assign(rre200b0_sum=t_LogRolling.Rain.mean())
X = X.assign(sre000b0_sum=t_LogRolling.Sunshine.mean())
X = X.assign(fu3010b0_mean=t_LogRolling.Wind.mean())
X = X.assign(prestab0_mean=t_LogRolling.Pressure.mean())

X = X.assign(log_sum = 0) #ToDo: Read Log and calculate the sum

X.transpose()


# ### Prepare DataSet (Poly, Scale)

# In[4]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from sklearn.preprocessing import PolynomialFeatures

polyDegree = 2
poly = PolynomialFeatures(degree=polyDegree)
X = poly.fit_transform(X).astype(int)

from sklearn.externals import joblib
X_min_max_scaler = joblib.load('data/X_min_max_scaler.pkl')
X = X_min_max_scaler.transform(X.reshape(1,-1))


# ## Predict!
# - 0 - means a lot of water
# - 1 - is medium
# - 2 - device should stay off

# In[5]:

#if result == 1:
#    model_reg = joblib.load('data/linreg_med_5deg.pkl')
#    res = model_reg.predict(X)[0]
#else:
#    model_reg = joblib.load('data/linreg_med_5deg.pkl')
#    res = model_reg.predict(X)[0]
from sklearn.externals import joblib
model = joblib.load('data/myLinReg.pkl')
    
y = model.predict(X)

y_min_max_scaler = joblib.load('data/y_min_max_scaler.pkl')
y_scaled = y / y_min_max_scaler.scale_ + y_min_max_scaler.min_
y_scaled


# In[6]:


#K = 10
#np.argpartition(model.coeff__,-K)[-K:]


# Export result to JSON

# In[13]:

import json

print((y_scaled[0]))
#print(json.dumps(y_scaled, ensure_ascii=False))


# In[ ]:



