# payments-fraud-detection-ml-demo
 
The below demo setup instructions assume an OpenShift cluster with the Nvidia Device Plugin already installed.

Nvidia Container Runtime:
https://devblogs.nvidia.com/gpu-containers-runtime/

Device Plugin references:
https://github.com/kubernetes/community/blob/master/contributors/design-proposals/resource-management/device-plugin.md
https://blog.openshift.com/how-to-use-gpus-with-deviceplugin-in-openshift-3-10/
https://docs.openshift.com/container-platform/3.11/dev_guide/device_manager.html
https://docs.openshift.com/container-platform/3.11/dev_guide/device_plugins.html#using-device-plugins
https://github.com/NVIDIA/k8s-device-plugin


### Install Kubeflow

Kubeflow on OpenShift installation steps have been adapted from the following tutorials:
https://www.lightbend.com/blog/how-to-deploy-kubeflow-on-lightbend-platform-openshift-introduction
https://www.kubeflow.org/docs/started/getting-started/
https://www.kubeflow.org/docs/gke/gcp-e2e/

# Install ksonnet

Kubeflow is installed using ksonnet, a templating framework for managing Kubernetes deployments.

Refer to the ks installation instructions for your environment here: https://ksonnet.io/get-started/

More detail on Kubeflow and ksonnet can be found at: https://www.kubeflow.org/docs/components/ksonnet/

# Install Kubeflow on OpenShift

First, oc login to your cluster, and ensure that you have cluster-admin privileges.
```
$ oc login <cluster url> -u <username>
$ oc get rolebindings | grep <username>

admin                                                    /admin                                 <username>
```

Create and navigate to a directory for Kubeflow. Download and generate the installation scripts for Kubeflow:
```
$ mkdir <kubeflow home>
$ cd <kubeflow home>
$ export KUBEFLOW_SRC=kubeflow
$ mkdir ${KUBEFLOW_SRC}
$ cd ${KUBEFLOW_SRC}
$ export KUBEFLOW_TAG=v0.4.1
$ curl https://raw.githubusercontent.com/kubeflow/kubeflow/${KUBEFLOW_TAG}/scripts/download.sh | bash
$ export KFAPP=openshift
$ scripts/kfctl.sh init ${KFAPP} --platform none
$ cd ${KFAPP}
$ ../scripts/kfctl.sh generate k8s
```

Permissions for Kubeflow components:
```
$ oc adm policy add-scc-to-user anyuid -z ambassador -n kubeflow
$ oc adm policy add-scc-to-user anyuid -z jupyter -n kubeflow
$ oc adm policy add-scc-to-user anyuid -z katib-ui -n kubeflow
$ oc adm policy add-scc-to-user anyuid -z default -n kubeflow
```

Update the TFJob operator to latest:
```
$ cd ks_app
$ ks param set tf-job-operator tfJobImage gcr.io/kubeflow-images-public/tf_operator:latest
```

Open the kfctl installation file for editing:
```
$ vi ../../scripts/kfctl.sh
```

To simplify installation of the demo, comment out components which will require persistent volumes:
```
ks apply default -c ambassador
ks apply default -c jupyter
ks apply default -c centraldashboard
ks apply default -c tf-job-operator
ks apply default -c pytorch-operator
ks apply default -c metacontroller
ks apply default -c spartakus
ks apply default -c argo
# ks apply default -c pipeline
```

```
# Reduce resource demands locally
# if [ "${PLATFORM}" != "minikube" ] && [ "${PLATFORM}" != "docker-for-desktop" ]; then
#  ks apply default -c katib
#fi
```

Save and close kfctl.sh, then run the Kubeflow installation:
```
$ cd ..
$ ../scripts/kfctl.sh apply k8s
```

Change to the new kubeflow project on your cluster and confirm all pods are running:
```
$ oc project kubeflow

$ oc get pods
NAME                                      READY     STATUS    RESTARTS   AGE
ambassador-5cf8cd97d5-26bbn               1/1       Running   0          2m
ambassador-5cf8cd97d5-k8rkn               1/1       Running   0          2m
ambassador-5cf8cd97d5-x8gfd               1/1       Running   0          2m
argo-ui-7c9c69d464-lsvv9                  1/1       Running   0          1m
centraldashboard-6f47d694bd-5zbwv         1/1       Running   0          1m
jupyter-0                                 1/1       Running   0          1m
metacontroller-0                          1/1       Running   0          1m
pytorch-operator-6f87db67b7-k75tv         1/1       Running   0          1m
spartakus-volunteer-d58c7444-z7l8q        1/1       Running   0          1m
tf-job-dashboard-7d9b99c66c-8gtnc         1/1       Running   0          1m
tf-job-operator-v1beta1-d6956688c-6mwbf   1/1       Running   0          1m
workflow-controller-5c95f95f58-bsf4m      1/1       Running   0          1m
```

Expose the ambassador service:
```
$ oc get svc
NAME               TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
ambassador         ClusterIP   172.30.38.58     <none>        80/TCP         29m
ambassador-admin   ClusterIP   172.30.33.69     <none>        8877/TCP       29m
argo-ui            NodePort    172.30.135.123   <none>        80:32164/TCP   28m
centraldashboard   ClusterIP   172.30.0.108     <none>        80/TCP         29m
jupyter-0          ClusterIP   None             <none>        8000/TCP       29m
jupyter-lb         ClusterIP   172.30.101.65    <none>        80/TCP         29m
tf-job-dashboard   ClusterIP   172.30.69.175    <none>        80/TCP         29m

$ oc expose svc/ambassador
route.route.openshift.io/ambassador exposed

$ oc get routes
NAME         HOST/PORT                                             PATH      SERVICES     PORT         TERMINATION   WILDCARD
ambassador   ambassador-kubeflow.apps.24a3.openshiftworkshop.com             ambassador   ambassador                 None
```

If/when needed, Kubeflow can be removed from the cluster by running the same template with the delete command:
```
../scripts/kfctl.sh delete k8s
```

### Upload Training Data to Cloud Storage and Create Buckets for Data and Model

This demo uses S3 as backend storage for the training data and the trained model. MinIO or a different object store can be substituted.

Make a bucket to store the training data and upload the file model-training/labeled-training-data/banksim_data_full.csv

For more information on the data set, see:
https://www.kaggle.com/ntnu-testimon/banksim1#bs140513_032310.csv


Create a bucket to store the trained model, and inside of the bucket create a folder called 'versions'.


### Create CPU and GPU Container Images for TensorFlow Training Job

Update both CPU and GPU training dockerfiles to use your training data and trained model bucket names:
```
$ vi model-training/cpu-training-container/Dockerfile

FROM tensorflow/tensorflow:1.13.1-py3
RUN pip install pandas
RUN pip install sklearn
RUN pip install boto3
RUN pip install tensorflow

ADD ../tensorflow-training-script/payment_fraud_training.py /opt/payment_fraud_training.py
RUN chmod +x /opt/payment_fraud_training.py

ENTRYPOINT ["/usr/bin/python3"]
CMD ["/opt/payment_fraud_training.py", "<training data bucket>", "<trained model bucket>"]
```

```
$ vi model-training/gpu-training-container/Dockerfile

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
CMD ["/opt/pay_fraud_training.py", "<training data bucket>", "<trained model bucket>"]
```

Update the TensorFlow training script to add your S3 credentials:
```
$ vi tensorflow-training-script/payment_fraud_training.py
# AWS S3 credentials and bucket names
# Add valid S3 credentials below
aws_id = "<AWS id>"
aws_key = "<AWS key>"
```

Create repositories in your container registry for the CPU and GPU training images, then build, tag and push:
```
$ docker login
```
```
$ cd model-training/cpu-training-container
$ export WORKING_DIR=$(pwd)
$ export VERSION_TAG=$(date +%s)
$ export DOCKER_REPO=<cpu training image repo>
$ export TRAIN_IMG_PATH=${DOCKER_REPO}:${VERSION_TAG}

$ docker build ${WORKING_DIR} -t ${TRAIN_IMG_PATH} -f Dockerfile
$ docker tag ${TRAIN_IMG_PATH} ${DOCKER_REPO}:${VERSION_TAG}
$ docker push ${DOCKER_REPO}:${VERSION_TAG}
```
```
$ cd model-training/gpu-training-container
$ export WORKING_DIR=$(pwd)
$ export VERSION_TAG=$(date +%s)
$ export DOCKER_REPO=<gpu training image repo>
$ export TRAIN_IMG_PATH=${DOCKER_REPO}:${VERSION_TAG}

$ docker build ${WORKING_DIR} -t ${TRAIN_IMG_PATH} -f Dockerfile
$ docker tag ${TRAIN_IMG_PATH} ${DOCKER_REPO}:${VERSION_TAG}
$ docker push ${DOCKER_REPO}:${VERSION_TAG}
```

-------------

### Define TFJob Templates for CPU and GPU Training Jobs

In your Kubeflow home directory, change to the ks_app project folder.
```
$ cd <kubeflow dir>/kubeflow/openshift/ks_app
```

Create and apply rolebinding for tf-jobs:
```
kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: tfjobs-role
  labels:
    app: tfjobs  
rules:
- apiGroups: ["kubeflow.org"]
  resources: ["tfjobs", "tfjobs/finalizers"]
  verbs: ["get", "list", "watch", "update", "patch"]
---
kind: RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: tfjobs-rolebinding
  labels:
    app: tfjobs  
roleRef:
  kind: Role
  name: tfjobs-role
subjects:
  - kind: ServiceAccount
    name: tf-job-operator
```
```
$ oc apply -f tfjobs-role.yaml -n kubeflow
```

Generate the CPU training job template:
```
$ export TF_JOB_NAME_CPU=pay-fraud-training-job-cpu
$ ks generate tf-job-simple-v1beta1 ${TF_JOB_NAME_CPU} --name=${TF_JOB_NAME_CPU}
```

Edit the template to use the CPU training image and define a TFJob which requests one CPU:
```
$ vi components/${TF_JOB_NAME_CPU}.jsonnet

local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["pay-fraud-training-job-cpu"];

local k = import "k.libsonnet";

local name = params.name;
local namespace = env.namespace;
local image = "<cpu training container image location>";

local tfjob = {
  apiVersion: "kubeflow.org/v1beta1",
  kind: "TFJob",
  metadata: {
    name: name,
    namespace: namespace,
  },
  spec: {
    tfReplicaSpecs: {
      Worker: {
        replicas: 1,
        template: {
          spec: {
            containers: [
              {
                image: image,
                name: "tensorflow",
                workingDir: "/opt/pay-fraud-training-job-cpu/scripts/pay_fraud",
                resources: {
                  limits:{
                    cpu: 1
                  },
                  requests: {
                    cpu: 1
                  }
                }
              },
            ],
            restartPolicy: "OnFailure",
          },
        },
      },
    },
  },
};

k.core.v1.list.new([
  tfjob,
])
```

Generate the GPU training job template:
```
$ export TF_JOB_NAME_GPU=pay-fraud-training-job-gpu
$ ks generate tf-job-simple-v1beta1 ${TF_JOB_NAME_GPU} --name=${TF_JOB_NAME_GPU}
```

Edit the template to use the GPU training image and define a TFJob which requests an Nvidia GPU:
```
$ vi components/${TF_JOB_NAME_GPU}.jsonnet

local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["pay-fraud-training-job-gpu"];

local k = import "k.libsonnet";

local name = params.name;
local namespace = env.namespace;
local image = "<gpu training container image location>";

local tfjob = {
  apiVersion: "kubeflow.org/v1beta1",
  kind: "TFJob",
  metadata: {
    name: name,
    namespace: namespace,
  },
  spec: {
    tfReplicaSpecs: {
      Worker: {
        replicas: 1,
        template: {
          spec: {
            containers: [
              {
                image: image,
                name: "tensorflow",
                workingDir: "/opt/pay-fraud-training-job-gpu/scripts/pay_fraud",
                resources: {
                  limits:{
                    "nvidia.com/gpu": 1
                  },
                 requests: {
                    "nvidia.com/gpu": 1
                  }
                }
              },
            ],
            restartPolicy: "OnFailure",
          },
        },
      },
    },
  },
};

k.core.v1.list.new([
  tfjob,
])

```


## Run TensorFlow Training Jobs on CPU and GPU Containers

Apply the CPU and GPU job templates to run the training scripts on the OpenShift cluster:
```
$ export KF_ENV=default
$ ks apply ${KF_ENV} -c ${TF_JOB_NAME_CPU}
$ ks apply ${KF_ENV} -c ${TF_JOB_NAME_GPU}
```


## Create TensorFlow Serving Job

Create and apply a secret to store AWS credentials:
```
$ vi awsaccess.yaml

apiVersion: v1
kind: Secret
metadata:
  name: awsaccess
data:
  AWS_ACCESS_KEY_ID: <AWS id>
  AWS_SECRET_ACCESS_KEY: <AWS key>
```
```
$ oc apply -f awsaccess.yaml -n kubeflow
```

In the ks_app directory, generate and apply the templates for the CPU TensorFlow serving service. Be sure to use your own trained model bucket location:
```
$ ks generate tf-serving-service payfraud-service-cpu
$ ks param set payfraud-service-cpu modelName payfraudcpu
$ ks param set payfraud-service-cpu trafficRule v1:100
$ ks param set payfraud-service-cpu serviceType LoadBalancer

$ ks generate tf-serving-deployment-aws payfraud-cpu-v1 --name=payfraudcpu
$ ks param set payfraud-cpu-v1 defaultCpuImage tensorflow/serving:1.12.0
$ ks param set payfraud-cpu-v1 defaultGpuImage tensorflow/serving:1.12.0-gpu
$ ks param set payfraud-cpu-v1 modelBasePath s3://<trained model bucket>/versions
$ ks param set payfraud-cpu-v1 s3Enable true
$ ks param set payfraud-cpu-v1 s3SecretName awsaccess
$ ks param set payfraud-cpu-v1 s3AwsRegion us-east-1
$ ks param set payfraud-cpu-v1 s3UseHttps 1
$ ks param set payfraud-cpu-v1 s3VerifySsl 0
$ ks param set payfraud-cpu-v1 s3Endpoint s3.us-east-1.amazonaws.com

$ ks apply default -c payfraud-cpu-v1
$ ks apply default -c payfraud-service-cpu
```

Do the same for the GPU TensorFlow serving service, again substituting your own trained model bucket location:
```
$ ks generate tf-serving-service payfraud-service-gpu
$ ks param set payfraud-service-gpu modelName payfraudgpu
$ ks param set payfraud-service-gpu trafficRule v1:100
$ ks param set payfraud-service-gpu serviceType LoadBalancer

$ ks generate tf-serving-deployment-aws payfraud-gpu-v1 --name=payfraudgpu
$ ks param set payfraud-gpu-v1 defaultCpuImage tensorflow/serving:1.12.0
$ ks param set payfraud-gpu-v1 defaultGpuImage tensorflow/serving:1.12.0-gpu
$ ks param set payfraud-gpu-v1 modelBasePath s3://<trained model bucket>/versions
$ ks param set payfraud-gpu-v1 s3Enable true
$ ks param set payfraud-gpu-v1 s3SecretName awsaccess
$ ks param set payfraud-gpu-v1 s3AwsRegion us-east-1
$ ks param set payfraud-gpu-v1 s3UseHttps 1
$ ks param set payfraud-gpu-v1 s3VerifySsl 0
$ ks param set payfraud-gpu-v1 s3Endpoint s3.us-east-1.amazonaws.com
$ ks param set payfraud-gpu-v1 numGpus 1

$ ks apply default -c payfraud-gpu-v1
$ ks apply default -c payfraud-service-gpu
```

If/when needed, the services can be deleted:
```
$ ks delete default -c payfraud-cpu-v1
$ ks delete default -c payfraud-service-cpu

$ ks delete default -c payfraud-gpu-v1
$ ks delete default -c payfraud-service-gpu
```

Expose routes for both services. We are using REST in this case, so expose port 8500/TCP for both services.

Confirm that the services are accessible and serving as expected. By default, the most recent version of the trained model in the object store will be served, but this can be adjusted.


# Service Status

Request:
GET <CPU serving route>/v1/models/payfraudcpu

Response:
```
{
    "model_version_status": [
        {
            "version": "1557304642",
            "state": "AVAILABLE",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        }
    ]
}
```

Request:
GET <GPU serving route>/v1/models/payfraudgpu

Response:
```
{
    "model_version_status": [
        {
            "version": "1557349851",
            "state": "AVAILABLE",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        }
    ]
}
```


# Model Metadata

Request:
GET <CPU serving route>/v1/models/payfraudcpu/versions/1557304642/metadata

Response:
```
{
    "model_spec": {
        "name": "payfraudcpu",
        "signature_name": "",
        "version": "1557304642"
    },
    "metadata": {
        "signature_def": {
            "signature_def": {
                "serving_default": {
                    "inputs": {
                        "payments": {
                            "dtype": "DT_FLOAT",
                            "tensor_shape": {
                                "dim": [
                                    {
                                        "size": "-1",
                                        "name": ""
                                    },
                                    {
                                        "size": "28",
                                        "name": ""
                                    }
                                ],
                                "unknown_rank": false
                            },
                            "name": "X_train:0"
                        }
                    },
                    "outputs": {
                        "classification": {
                            "dtype": "DT_FLOAT",
                            "tensor_shape": {
                                "dim": [
                                    {
                                        "size": "-1",
                                        "name": ""
                                    },
                                    {
                                        "size": "2",
                                        "name": ""
                                    }
                                ],
                                "unknown_rank": false
                            },
                            "name": "Softmax:0"
                        }
                    },
                    "method_name": "tensorflow/serving/predict"
                }
            }
        }
    }
}
```

Request:
GET <GPU serving route>/v1/models/payfraudgpu/versions/1557349851/metadata

Response:
```
{
    "model_spec": {
        "name": "payfraudgpu",
        "signature_name": "",
        "version": "1557349851"
    },
    "metadata": {
        "signature_def": {
            "signature_def": {
                "serving_default": {
                    "inputs": {
                        "payments": {
                            "dtype": "DT_FLOAT",
                            "tensor_shape": {
                                "dim": [
                                    {
                                        "size": "-1",
                                        "name": ""
                                    },
                                    {
                                        "size": "28",
                                        "name": ""
                                    }
                                ],
                                "unknown_rank": false
                            },
                            "name": "X_train:0"
                        }
                    },
                    "outputs": {
                        "classification": {
                            "dtype": "DT_FLOAT",
                            "tensor_shape": {
                                "dim": [
                                    {
                                        "size": "-1",
                                        "name": ""
                                    },
                                    {
                                        "size": "2",
                                        "name": ""
                                    }
                                ],
                                "unknown_rank": false
                            },
                            "name": "Softmax:0"
                        }
                    },
                    "method_name": "tensorflow/serving/predict"
                }
            }
        }
    }
}
```


# Prediction Service

The below predictions requests are for fraudulent payments. The output gives the predicted value of non-fraud, and then the predicted value for the payment to be fraud.

Request:
POST <CPU serving route>/v1/models/payfraudcpu:predict
```
{
  "inputs": [[0.005313,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000]]
}
```

Response:
```
{
    "outputs": [
        [
            0.016938,
            0.983062
        ]
    ]
}
```

Request:
POST <GPU serving route>/v1/models/payfraudgpu:predict
```
{
  "inputs": [[0.005313,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000]]
}
```

Response:
```
{
    "outputs": [
        [
            0.0109494,
            0.989051
        ]
    ]
}
```
