FROM tensorflow/tensorflow:1.13.1-gpu-py3
RUN pip install pandas
RUN pip install sklearn
RUN pip install tensorflow-gpu
RUN pip install boto3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=5.0"

ADD ../tensorflow-training-script/payment_fraud_training.py /opt/pay_fraud_training.py
RUN chmod +x /opt/pay_fraud_training.py

ENTRYPOINT ["/usr/bin/python3"]
CMD ["/opt/pay_fraud_training.py", "pay-fraud-demo-data", "pay-fraud-demo-model"]
