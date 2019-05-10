import sys
import os, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import boto3
import time
import calendar
from sklearn.preprocessing import OneHotEncoder

####
# This script uses the synthentic payments fraud research data set presented by Edgar Lopez et al:
# - https://www.kaggle.com/ntnu-testimon/banksim1
# - https://www.researchgate.net/publication/265736405_BankSim_A_Bank_Payment_Simulation_for_Fraud_Detection_Research
#
# The training method follows the approach described at:
# - https://www.kaggle.com/raymondxie/credit-card-fraud-detection-with-tensorflow
#
# Using Amazon S3 storage, the script:
# - Downloads labeled data csv from S3
# - Performs data cleanup, one hot encoding, separation into feature/label training/test data sets
# - Trains and tests the model
# - Uploads timestamp-versioned trained model to an S3 bucket
#
####

# AWS S3 credentials and bucket names
# Add valid S3 credentials below
aws_id = ""
aws_key = ""
s3_data_bucket_name = sys.argv[1]
s3_model_bucket_name = sys.argv[2]

export_dir="/opt/saved_model"
ts = time.time()
ts = calendar.timegm(time.gmtime())
ts_str = str( ts )
export_path = os.path.join(tf.compat.as_bytes(export_dir), tf.compat.as_bytes(ts_str))
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

## Read in full dataset
print("\n## Reading Original Data")
data = pd.read_csv('https://s3.amazonaws.com/' + s3_data_bucket_name + '/banksim_data_full.csv')
print("Data Shape:", data.shape)

## Print head of data
#print(data.head())

number_frauds = len(data[data['fraud'] == 1])
print("\nNumber of frauds in data set: ", number_frauds)

number_non_frauds = len(data[data['fraud'] == 0])
print("Number of non-frauds in data set: ", number_non_frauds)

print("Total number of transactions: ", len(data.index))
print(number_non_frauds + number_frauds)

fraud_ratio = float(number_frauds / (number_non_frauds + number_frauds))
print("Ratio of fraud: ", fraud_ratio)

print("\n\n============================================================================")
print("## Cleaning and Formatting Data")

# Selecting data columns to train on
data = data[['fraud','amount','category','age','gender']]
# One-hot encoding
data = pd.get_dummies(data)
data = pd.get_dummies(data, columns=['fraud'])
#print(data.head())
#print(type(data))
#print(data.loc[0:])

normalized_data = (data - data.min()) / (data.max() - data.min())
print("\nExample payment records:")
# A non-fraud payment:
print("Non-fraud Payment")
print(normalized_data.loc[0,:])

print("\n")
# Finding the index for a fraud record
#frauds = normalized_data[normalized_data.fraud_1 == 1.000000]
#print(frauds.head)
# A fraud payment:
print("Fraud Payment")
print(normalized_data.loc[88,:])

# Separate features into X dataframe
dataframe_X = normalized_data.drop(['fraud_0', 'fraud_1'], axis=1)
#print(dataframe_X[:5])

# Separate labels to y dataframe
dataframe_y = normalized_data[['fraud_0', 'fraud_1']]
#print(dataframe_y[:5])

nparray_X, nparray_y = np.asarray(dataframe_X.values, dtype='float32'), np.asarray(dataframe_y.values, dtype='float32')

# Separate training and test records 80/20
train_size = int(0.8 * len(nparray_X))
(raw_X_train, raw_y_train) = (nparray_X[:train_size], nparray_y[:train_size])
(raw_X_test, raw_y_test) = (nparray_X[train_size:], nparray_y[train_size:])

weighting = 1 / fraud_ratio
raw_y_train[:, 1] = raw_y_train[:, 1] * weighting

input_dimensions = nparray_X.shape[1]
print(input_dimensions)

# 2 cells for the output
output_dimensions = nparray_y.shape[1]
print(output_dimensions)

# 100 cells for the 1st layer
num_layer_1_cells = 100
# 150 cells for the second layer
num_layer_2_cells = 150

# We will use these as inputs to the model when it comes time to train it (assign values at run time)
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='y_train')

# We will use these as inputs to the model once it comes time to test it
X_test_node = tf.constant(raw_X_test, name='X_test')
y_test_node = tf.constant(raw_y_test, name='y_test')

# First layer takes in input and passes output to 2nd layer
weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name='weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name='biases_1')

# Second layer takes in input from 1st layer and passes output to 3rd layer
weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]), name='weight_2')
biases_2_node = tf.Variable(tf.zeros([num_layer_2_cells]), name='biases_2')

# Third layer takes in input from 2nd layer and outputs [1 0] or [0 1] depending on fraud vs legit
weight_3_node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]), name='weight_3')
biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name='biases_3')

def network(input_tensor):
    # Sigmoid fits modified data well
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
    # Dropout prevents model from becoming lazy and over confident
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_node) + biases_2_node), 0.85)
    # Softmax works very well with one hot encoding which is how results are outputted
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3_node) + biases_3_node)
    return layer3

y_train_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)

cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_prediction)

optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return (100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0])

num_epochs = 1


print("\n\n============================================================================")
print("## Training Model")

# Start Timer
start = time.time()
print("\nStarting timer at: " + str(start))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        start_time = time.time()

        _, cross_entropy_score = session.run([optimizer, cross_entropy],feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})

        # if epoch % 10 == 0:
        timer = time.time() - start_time

        print('Epoch: {}'.format(epoch), 'Current loss: {0:.4f}'.format(cross_entropy_score),
              'Elapsed time: {0:.2f} seconds'.format(timer))

        final_y_test = y_test_node.eval()
        final_y_test_prediction = y_test_prediction.eval()
        final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
        print("Current accuracy: {0:.2f}%".format(final_accuracy))

    # Record end time
    end = time.time()
    print("\nTraining end time: " + str(end))
    print("Training completed in " + str(end - start) + " seconds\n")

    tensor_info_input = tf.saved_model.utils.build_tensor_info(X_train_node)
    tensor_info_output = tf.saved_model.utils.build_tensor_info(y_train_prediction)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'payments': tensor_info_input},
            outputs={'classification': tensor_info_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        })

    # export the model
    builder.save()
    #print('Done exporting!')


# Upload trained model to S3 bucket
bucket = s3_bucket_name
print("\nUploading trained model to remote store:", bucket)
print("Version: ", ts_str)
print("\n\n")

file_name = "/opt/saved_model/" + ts_str + "/saved_model.pb"
object_name = "versions/" + ts_str + "/saved_model.pb"

variables_data_name = "/opt/saved_model/" + ts_str + "/variables/variables.data-00000-of-00001"
variables_data_object_name = "versions/" + ts_str + "/variables/variables.data-00000-of-00001"

variables_index_name = "/opt/saved_model/" + ts_str + "/variables/variables.index"
variables_index_object_name = "versions/" + ts_str + "/variables/variables.index"

session = boto3.Session(aws_access_key_id=aws_id, aws_secret_access_key=aws_key)
s3_client = session.client('s3')

s3_client.upload_file(file_name, bucket, object_name)
s3_client.upload_file(variables_data_name, bucket, variables_data_object_name)
s3_client.upload_file(variables_index_name, bucket, variables_index_object_name)
