FROM tensorflow/tensorflow:1.13.1-py3
RUN pip install pandas
RUN pip install sklearn
RUN pip install boto3
RUN pip install tensorflow

ADD ../tensorflow-training-script/payment_fraud_training.py /opt/payment_fraud_training.py
RUN chmod +x /opt/payment_fraud_training.py

ENTRYPOINT ["/usr/bin/python3"]
CMD ["/opt/payment_fraud_training.py", "pay-fraud-demo-data", "pay-fraud-demo-model"]
