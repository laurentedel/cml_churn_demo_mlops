## Explained Model Serving
# The `explain` function will load the trained model and calculate a new explained 
# prediction from a single data row.

from collections import ChainMap
import cdsw, numpy
from churnexplainer import ExplainedModel

#Load the model save earlier.
em = ExplainedModel(model_name='telco_linear',data_dir='/home/cdsw')

# *Note:* If you want to test this in a session, comment out the line `@cdsw.model_metrics` below. Don't forget to uncomment when you
# deploy, or it won't write the metrics to the database 
@cdsw.model_metrics
# This is the main function used for serving the model. It will take in the JSON formatted arguments , calculate the probablity of 
# churn and create a LIME explainer explained instance and return that as JSON.
def explain(args):
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    
    #NEW! Track our inputs
#    for key in data:
#      if isinstance(data[key], numpy.int64) or isinstance(data[key], numpy.float64):
#        cdsw.track_metric(key, data[key].item())
#      else:
#        cdsw.track_metric(key, data[key])

    cdsw.track_metric('input_data', data)
    
    #NEW! Track our prediction
    cdsw.track_metric('probability', probability)
    
    #NEW! Track explanation
    cdsw.track_metric('explanation', explanation)
    
    return {
        'data': dict(data),
        'probability': probability,
        'explanation': explanation
        }

#To test this is a session, uncomment and run the two rows below.
#x={"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}
#explain(x)
