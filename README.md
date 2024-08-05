# Classification-on-aws
Upload,train,test,deploy,Predict and delete data on aws

This code is an end-to-end machine learning workflow in AWS using Amazon SageMaker and Boto3 for training and deploying an XGBoost model on a dataset, and making predictions. Here is an explanation of each section:

Imports and Initial Setup
python
Copy code
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session
import pandas as pd
import urllib
import os
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
import numpy as np
Imports: Necessary libraries for data handling (Pandas, Numpy), AWS services (Boto3, SageMaker), and other utilities.
S3 Bucket Creation
python
Copy code
bucket_name = 'bankapplication-project' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
my_region = boto3.session.Session().region_name # set the region of the instance
print(my_region)

s3 = boto3.resource('s3')
my_region = 'ap-south-1'  # Replace with your desired region

try:
    if my_region == 'ap-south-1':
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': my_region}
        )
    else:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': my_region}
        )
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ', e)
Bucket Creation: Creates an S3 bucket to store data and model artifacts, handling region-specific constraints.
Downloading and Loading Data
python
Copy code
prefix = 'xgboost-formyproject'
output_path = 's3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)

try:
    urllib.request.urlretrieve("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ', e)

try:
    model_data = pd.read_csv('./bank_clean.csv', index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ', e)
Downloading Data: Downloads a dataset from a URL and loads it into a Pandas DataFrame.
Splitting Data
python
Copy code
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
Splitting Data: Randomly splits the data into training (70%) and testing (30%) sets.
Uploading Data to S3
python
Copy code
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = TrainingInput(s3_data=f's3://{bucket_name}/{prefix}/train/train.csv', content_type='csv')

test_data_to_save = pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1)
test_data_to_save.to_csv('test.csv', index=False, header=False)
s3 = boto3.Session().resource('s3')
s3.Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = TrainingInput(s3_data=f's3://{bucket_name}/{prefix}/test/test.csv', content_type='text/csv')

print('Test data saved to S3 and TrainingInput defined.')
Saving and Uploading Data: Converts training and testing data to CSV files and uploads them to S3.
Model Training
python
Copy code
container = sagemaker.image_uris.retrieve("xgboost", sagemaker.Session().boto_region_name, version="1.3-1")
hyperparameters = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "objective": "binary:logistic",
    "num_round": 50
}

estimator = Estimator(image_uri=container, 
                      hyperparameters=hyperparameters,
                      role=sagemaker.get_execution_role(),
                      train_instance_count=1, 
                      train_instance_type='ml.m5.2xlarge', 
                      train_volume_size=5, # 5 GB 
                      output_path=output_path,
                      train_use_spot_instances=True,
                      train_max_run=300,
                      train_max_wait=600)

estimator.fit({'train': s3_input_train, 'validation': s3_input_test})
Model Training: Defines and trains an XGBoost model using the training data. The model is trained on a specified instance type and uses spot instances to save costs.
Model Deployment and Predictions
python
Copy code
xgb_predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge', serializer=CSVSerializer())

test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values  # Load the data into an array

xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = CSVSerializer()

predictions = xgb_predictor.predict(test_data_array).decode('utf-8')

print("Raw predictions:", predictions)

try:
    predictions_list = [float(pred) for pred in predictions.strip().split(',')]
    predictions_array = np.array(predictions_list)
    
    print(predictions_array.shape)
    print(predictions_array)
except ValueError as e:
    print("Error converting predictions to floats:", e)
Model Deployment: Deploys the trained model to an endpoint for making predictions.
Predictions: Makes predictions on the test data and handles the prediction output, converting it to a NumPy array.
Evaluation and Cleanup
python
Copy code
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])

tn = cm.iloc[0, 0] if 0 in cm.columns and 0 in cm.index else 0
fn = cm.iloc[1, 0] if 0 in cm.columns and 1 in cm.index else 0
tp = cm.iloc[1, 1] if 1 in cm.columns and 1 in cm.index else 0
fp = cm.iloc[0, 1] if 1 in cm.columns and 0 in cm.index else 0

total = tp + tn + fp + fn
p = (tp + tn) / total * 100 if total != 0 else 0

print("\n{0:<30}{1:.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>10}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.1f}% ({2:<6}){3:>8.1f}% ({4:<6})".format("No Purchase", tn / (tn + fn) * 100 if (tn + fn) != 0 else 0, tn, fp / (tp + fp) * 100 if (tp + fp) != 0 else 0, fp))
print("{0:<15}{1:<2.1f}% ({2:<6}){3:>8.1f}% ({4:<6})\n".format("Purchase", fn / (tn + fn) * 100 if (tn + fn) != 0 else 0, fn, tp / (tp + fp) * 100 if (tp + fp) != 0 else 0, tp))
Confusion Matrix: Creates and prints a confusion matrix to evaluate the model's performance.
Classification Rate: Calculates and prints the overall classification rate and other metrics.
Cleanup
python
Copy code
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
Cleanup: Deletes the deployed endpoint and all objects in the S3 bucket to clean up resources.
This code demonstrates the entire process of setting up an S3 bucket, downloading and processing data, training a model, making predictions, evaluating the model, and cleaning up resources in AWS using SageMaker and Boto3.
