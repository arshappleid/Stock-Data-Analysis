import pandas as pd;
import numpy as np;
from neural_net import neural_net

data = pd.read_csv('./../RAW_DATA/JPMorgan Chase.csv');
train_data = data.iloc[:-1]
latest_observation = data.iloc[-1]
latest_observation = latest_observation[['Open','High','Low']] ## In real world we will only have these values
latest_observation = np.array(latest_observation).reshape(-1,1);
actual_value = data.iloc[-1][['Close']]

neural_net_model = neural_net(train_data , latest_observation);

predicted_value = neural_net_model.getPredictedValue();


print(f"Actual Value : {actual_value}");
print(f"Predicted Value : {predicted_value}");

