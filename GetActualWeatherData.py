
# coding: utf-8

# # Download Weather Data from MeteoSuisse

# ## Download Data from MeteoSwiss

# In[40]:

import pandas as pd

data = pd.read_csv('http://data.geo.admin.ch/ch.meteoschweiz.swissmetnet/VQHA69.csv',
                   sep = '|',
                   header = 1)

data.head()


# ## Select Data only for Wädenswil (near Thalwil)

# In[41]:

data = data.loc[data.stn == 'WAE',:]


# ## Post Data to DataBase

# In[42]:

import pymysql.cursors

connection = pymysql.connect(host = "mikmak.cc", user="sensor", passwd="Gaffe2017", db="weatherDW")
#sql = ('SELECT MsgID, msgv1, TmStp FROM sysLog WHERE msgID = "109" AND sysID = "79cf6c22-dcc6-11e5-8e77-00113217113f"')

query = ("CALL `writeWeatherLog`("+ "'" + "79cf6c22-dcc6-11e5-8e77-00113217113f" + "'"  + ","+ "'" 
         + str(data.tre200s0.values[0]) + "'" + ","+ "'" 
         + str(data.fu3010z0.values[0]) + "'" + ","+ "'"  
         + str(data.rre150z0.values[0]) + "'"+ ","+ "'" 
         + str(data.prestas0.values[0]) + "'" + ","+ "'" 
         + str(data.ure200s0.values[0]) + "'" + ","+ "'" 
         + str(data.stn.values[0]) + "'" +",'" 
         + str(data.stn.values[0]) + "'" +",'" 
         + str(data.sre000z0.values[0]) +  "'" + ")")

with connection.cursor() as cursor:
    cursor.execute(query)
connection.commit()
e_Log = cursor.fetchall()
connection.close()
e_Log

print("OK")


# Stored Procedure executed on DataBase:
# 
# 
# ``` sql
# BEGIN
# 
# DECLARE logID varchar(36);
# SELECT UUID() INTO logID;
# INSERT INTO log
# 	(
#      sysID	,
#      logID	,
#      msgID	,
#      msgv1
#     )
# VALUES 
# 	(
#      sysID		,
#      logID		,
#      '801'	,
#      temperature
#     ) ;  
# INSERT INTO log	(sysID	, logID	,     msgID	,     msgv1    )
# VALUES 	(
#      sysID		,     logID		,     '802'	,     pressure
#     ) ;
# INSERT INTO log	(sysID	, logID	,     msgID	,     msgv1    )
# VALUES 	(
#      sysID		,     logID		,     '820'	,     humidity
#     ) ;
# INSERT INTO log	(sysID	, logID	,     msgID	,     msgv1    )
# VALUES 	(
#      sysID		,     logID		,     '841'	,     wind
#     ) ;
# INSERT INTO log	(sysID	, logID	,     msgID	,     msgv1    )
# VALUES 	(
#      sysID		,     logID		,     '851'	,     rain
#     ) ;
# INSERT INTO log	(sysID	, logID	,     msgID	,     msgv1    )
# VALUES 	(
#      sysID		,     logID		,     '101'	,     lon
#     ) ;
# INSERT INTO log	(sysID	, logID	,     msgID	,     msgv1    )
# VALUES 	(
#      sysID		,     logID		,     '102'	,     lat
#     ) ;
# INSERT INTO log	(sysID	, logID	,     msgID	,     msgv1    )
# VALUES 	(
#      sysID		,     logID		,     '852'	,     sunshine
#     ) ;
# END
# ```

# In[ ]:



