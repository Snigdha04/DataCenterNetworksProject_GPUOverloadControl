# DataCenterNetworksProject_GPUOverloadControl

DISCLAIMER: I executed these instructions on a fresh cloudlab experiment on 2021-11-24.

## Chameleon Cloud Setup ##

**Step1:** Create a new lease in Chamelon cloud with following configuration:
``` Bash
resource type   physical:host
missing resources   No
resources changed   No
hypervisor_properties   -
resource_properties   ["==","$node_type","gpu_rtx_6000"]
before_end    default
on_start    default
min   2
max   2
```
**Step2:** Create/Add key pair under **Compute/Key Pairs** in dashboard

**Step3:** Initiate two instances under **Compute/Instances** in dashboard of type:
``` Bash
Image Name CC-Ubuntu20.04-CUDA11-20211129
```
Make sure to set your ssh key in the instance config

**Step4:** Create/Add two floating IPs under **Network/Floating IPs** in dashboard and attach them to the created instances

SSH to your cloudlab machine(s). For example, I use the following:
``` Bash
ssh -i .ssh/mychameloeonkey.pem cc@192.5.87.110
```


## TRITON Inference Engine Setup ##

Clone the Triton repository and install NVIDIA docker:
``` Bash
git clone https://github.com/triton-inference-server/server.git
curl https://get.docker.com | sh   && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo usermod -aG docker $USER
newgrp docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
``` 

Create an account in nvcr.io and login:
``` Bash
docker login nvcr.io
Username: $oauthtoken
Password: Z2I0Mm5xMTNyMTc1dWRuYmFsNWFkaGZybmo6ODAwNTNmNWEtYzZmNy00MTcwLTljYzUtNWFjMDVkM2FlMjlh
``` 

### If testing with MPS enabled ###

On the server node:
``` Bash
NVIDIA_DRIVER_CAPABILITIES=utility
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d
``` 

To quit MPS:
``` Bash
sudo echo quit | sudo nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT
``` 
### Setup on server node ###

#### Fetch the Triton image and Create A Model Repository ####
``` Bash
docker pull nvcr.io/nvidia/tritonserver:22.02-py3
cd server/docs/examples
./fetch_models.sh
```
#### Run Triton ####

##### Without MPS #####
Hosting model 1
``` Bash
sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ~/gpuoverload/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models
```
Hosting model 2
``` Bash
sudo docker run --gpus=1 --rm -p8003:8000 -p8004:8001 -p8005:8002 -v ~/gpuoverload/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models
```

##### With MPS #####
Hosting model 1
``` Bash
sudo docker run --gpus=1 --ipc=host --runtime=nvidia --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ~/gpuoverload/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models
```
Hosting model 2
``` Bash
sudo docker run --gpus=1 --ipc=host --runtime=nvidia --rm -p8003:8000 -p8004:8001 -p8005:8002 -v ~/gpuoverload/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models
```

To watch the GPU utilization:
``` Bash
watch -n 1 "curl -v --silent localhost:8002/metrics --stderr - | grep 'nv_gpu_utilization\|nv_gpu_power_usage'"
```

### Setup on client node ###

``` Bash
docker pull nvcr.io/nvidia/tritonserver:22.02-py3-sdk

sudo docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:22.02-py3-sdk

perf_analyzer -m densenet_onnx -u 192.5.87.195:8001 -i gRPC --async -p 1000 --request-rate-range 100:200:5 --request-distribution poisson -s 100 -b 1 -f stats.csv
```





