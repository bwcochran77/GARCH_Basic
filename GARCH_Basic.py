#Import Requirements-----------------------------------------------------------

from arch import arch_model
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
from yahoo_fin import stock_info as si

#Set Date Range----------------------------------------------------------------
start = dt.datetime(2000,1,1)
end = dt.datetime.now()
daterange = pd.date_range(start, end).tolist()
daterange=[str(i) for i in daterange]

#Set Stock List----------------------------------------------------------------

Stock = 'SPY'

#Get Data from Yahoo-----------------------------------------------------------
data = si.get_data(Stock,start_date = start, end_date = end)
print(data)
# Calculate % returns for dataset----------------------------------------------
returns = 100 * data['adjclose'].pct_change().dropna()
lastprice = data['adjclose'].tail(1)

#Setup and paramaterize GARCH model--------------------------------------------
am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='normal')

#Model Inputs------------------------------------------------------------------
pred_days = 2 #number of days from today for prediction vs training
fcast_length = 20 #number of days to predict into the future 

#Setup training set and prediction set-----------------------------------------
split_date = dt.datetime.today() - timedelta(days=pred_days)
res = am.fit(last_obs=split_date)

#Make Forecasts----------------------------------------------------------------
forecasts =(res.forecast(horizon=fcast_length, start=split_date, method='simulation'))

#Generate Forecast Output------------------------------------------------------

output = forecasts.variance #output forecast variance
volprediction = (output**.5)*15.87 #convert variance forecast to annualized volatility
volpredictionmean = volprediction.mean()



#Output Last n days of Prediction----------------------------------------------

print(volprediction.tail(1).to_string())
volprediction.tail(1).T.plot()

