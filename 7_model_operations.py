## Model Operations
# This script show cases how to use the model operations features of CML

import cdsw, time, os, random, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
import requests
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns


## Set the model ID
# Get the model id from the model you deployed in step 5. These are unique to each 
# model on CML.

model_id = "111"

# Grab the data from Hive.
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

df = spark.sql("SELECT * FROM default.telco_churn").toPandas()

# Get the various Model CRN details
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

Model_CRN = latest_model ["crn"]
Deployment_CRN = latest_model["latestModelDeployment"]["crn"]
model_endpoint = HOST.split("//")[0] + "//modelservice." + HOST.split("//")[1] + "/model"

# Get the row start number for where new model data will be added. Read in the model metics and check
# the 'mertics' key length.

model_metrics = cdsw.read_metrics(model_crn=Model_CRN,model_deployment_crn=Deployment_CRN)
if (model_metrics["metrics"] == None):
  metric_start_index = 0
else:
  metric_start_index = len(model_metrics["metrics"])
  
## Generate Sample Data
# This section will grab 750 random samples from the data set and simulate 250 predictions 
# per week. The live model will be called 750 times in 3 loops. Week 1 the data passes directly 
# to the model, for weeks 2 and 3, will introduce some errors into the actual churn 
# value, which will make the model less accurate. These accuracy measures are tracked 
# per week and plotted at the end.

# Note: Don't test the model while the loops are running. The accuracy measure goes
# out of sync.

# Randomly flip some "No" Churn Values to "Yes"
def flip_churn(item,percent):
  if random.random() < percent:
    return item.replace({'Churn':'No'},'Yes')
  else:
    return item

# Get 750 samples  
df_sample = df.sample(750)
record_count = 0

### Week 1

# Grab the first 250 rows
df_week_1 = df_sample.iloc[0:250,:]
df_week_1.groupby('Churn')['Churn'].count() 

# Clean up the data
df_week_1_clean = df_week_1.\
  replace({'SeniorCitizen': {"1": 'Yes', "0": 'No'}}).\
  replace(r'^\s$', np.nan, regex=True).\
  dropna()

# Get the actual label values
week_1_lables = (df_week_1_clean["Churn"] == 'Yes').values

# Drop the Churn column.
df_week_1_clean = df_week_1_clean.drop(['Churn','customerID'], axis=1)

# Create an array of model responses.
response_lables_week_1 = []

# Loop through the 500 records and call the model for each row. This will add the record
# of the call to the model metrics and adds the response to the response array.
for record in json.loads(df_week_1_clean.to_json(orient='records')):
  print("Processed {} records".format(record_count)) if (record_count%50 == 0) else None
  record_count+=1
  r = requests.post(
    model_endpoint, 
    data='{"accessKey":"' + latest_model["accessKey"] + '","request":' + json.dumps(record) +'}', 
    headers={'Content-Type': 'application/json'}
  )
  response_lables_week_1.append(r.json()["response"]["prediction"]["probability"] >= 0.5)

# Check the accuracy of the model responses.  
df_week_1_accuracy = classification_report(week_1_lables,response_lables_week_1,output_dict=True)["accuracy"]

# Show the info for cdsw.track_aggregate_metrics
help(cdsw.track_aggregate_metrics)

# Add the accuracy metric, along with start and end timestamps to the model metrics
w1_start_timestamp_ms = int(round(time.time() * 1000)) - 7*24*60*60*1000*3
w1_end_timestamp_ms = int(round(time.time() * 1000)) - 7*24*60*60*1000*2
cdsw.track_aggregate_metrics({"accuracy": df_week_1_accuracy}, w1_start_timestamp_ms , w1_end_timestamp_ms, model_deployment_crn=Deployment_CRN)

### Week 2
# Same as week 1, now with rows 251 - 500

df_week_2 = df_sample.iloc[251:500,:]

df_week_2 = df_week_2.apply(lambda x: flip_churn(x,0.2),axis=1)
df_week_2.groupby('Churn')['Churn'].count() 

df_week_2_clean = df_week_2.\
  replace({'SeniorCitizen': {"1": 'Yes', "0": 'No'}}).\
  replace(r'^\s$', np.nan, regex=True).\
  dropna()

week_2_lables = (df_week_2_clean["Churn"] == 'Yes').values

df_week_2_clean = df_week_2_clean.drop(['Churn','customerID'], axis=1)

response_lables_week_2 = []

for record in json.loads(df_week_2_clean.to_json(orient='records')):
  print("Processed {} records".format(record_count)) if (record_count%50 == 0) else None
  record_count+=1
  r = requests.post(
    model_endpoint, 
    data='{"accessKey":"' + latest_model["accessKey"] + '","request":' + json.dumps(record) +'}', 
    headers={'Content-Type': 'application/json'}
  )
  response_lables_week_2.append(r.json()["response"]["prediction"]["probability"] >= 0.5)
  
df_week_2_accuracy = classification_report(week_2_lables,response_lables_week_2,output_dict=True)["accuracy"]

w2_start_timestamp_ms = int(round(time.time() * 1000)) - 7*24*60*60*1000*2
w2_end_timestamp_ms = int(round(time.time() * 1000)) - 7*24*60*60*1000

cdsw.track_aggregate_metrics({"accuracy": df_week_2_accuracy}, w2_start_timestamp_ms , w2_end_timestamp_ms, model_deployment_crn=Deployment_CRN)
  
### Week 3
# Same as week 1, now with rows 501 - 750

df_week_3 = df_sample.iloc[501:750,:]

df_week_3 = df_week_3.apply(lambda x: flip_churn(x,0.4),axis=1)
df_week_3.groupby('Churn')['Churn'].count() 

df_week_3_clean = df_week_3.\
  replace({'SeniorCitizen': {"1": 'Yes', "0": 'No'}}).\
  replace(r'^\s$', np.nan, regex=True).\
  dropna()

week_3_lables = (df_week_3_clean["Churn"] == 'Yes').values

df_week_3_clean = df_week_3_clean.drop(['Churn','customerID'], axis=1)

response_lables_week_3 = []

for record in json.loads(df_week_3_clean.to_json(orient='records')):
  print("Processed {} records".format(record_count)) if (record_count%50 == 0) else None
  record_count+=1
  r = requests.post(
    model_endpoint, 
    data='{"accessKey":"' + latest_model["accessKey"] + '","request":' + json.dumps(record) +'}', 
    headers={'Content-Type': 'application/json'}
  )
  response_lables_week_3.append(r.json()["response"]["prediction"]["probability"] >= 0.5)
  
df_week_3_accuracy = classification_report(week_3_lables,response_lables_week_3,output_dict=True)["accuracy"]

w3_start_timestamp_ms = int(round(time.time() * 1000)) - 7*24*60*60*1000
w3_end_timestamp_ms = int(round(time.time() * 1000))

cdsw.track_aggregate_metrics({"accuracy": df_week_3_accuracy}, w3_start_timestamp_ms , w3_end_timestamp_ms, model_deployment_crn=Deployment_CRN)

## Plot some Model Metrics
# Here are some plots from the model metrics.  

# Read in the model metrics dict.
model_metrics = cdsw.read_metrics(model_crn=Model_CRN,model_deployment_crn=Deployment_CRN)

# This is a handy way to unravel the dict into a big pandas dataframe.
metrics_df = pd.io.json.json_normalize(model_metrics["metrics"][metric_start_index:])
metrics_df.tail().T

# Do some conversions & calculations
metrics_df['startTimeStampMs'] = pd.to_datetime(metrics_df['startTimeStampMs'], unit='ms')
metrics_df['endTimeStampMs'] = pd.to_datetime(metrics_df['endTimeStampMs'], unit='ms')
metrics_df["processing_time"] = (metrics_df["endTimeStampMs"] - metrics_df["startTimeStampMs"]).dt.microseconds * 1000

# This shows how to plot specific metrics.
sns.set_style("whitegrid")
sns.despine(left=True,bottom=True)

prob_metrics = metrics_df.dropna(subset=['metrics.probability'])
sns.lineplot(x=range(len(prob_metrics)), y="metrics.probability", data=prob_metrics, color='grey')

time_metrics = metrics_df.dropna(subset=['processing_time'])
sns.lineplot(x=range(len(prob_metrics)), y="processing_time", data=prob_metrics, color='grey')

# This shows how the model accuracy drops over time.
agg_metrics = metrics_df.dropna(subset=["metrics.accuracy"])
sns.barplot(x=list(range(1,len(agg_metrics)+1)), y="metrics.accuracy", color="grey", data=agg_metrics)
