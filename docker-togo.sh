# nvidio-docker tutorial
https://docs.nvidia.com/deeplearning/dgx/bp-docker/index.html

# check cuda version support for tensorflow
https://www.tensorflow.org/install/source#tested_source_configurations

# list all running containers
docker ps -a
## CONTAINER ID IMAGE COMMAND CREATED STATUS...

# re-entering a running docker
docker exec -it [container-id] bash
docker attach [container-id]

# pull and run a container i:interactive, t: tty, p: publish port
nvidia-docker run -it tensorflow/tensorflow:1.4.1-gpu  # 1.4.1 is the latest tensorflow that support cuda 8 but this does not work
docker pull nvcr.io/nvidia/tensorflow:19.03-py<x>  # might require docker login
nvidia-docker run -it tensorflow/tensorflow:1.13.1-py3 # always use the python3

# sign up to docker hub and nvidia ngc
# debug
make sure  "nvidia-smi" works
# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
docker rmi -f <image ID> and rerun command  # does not work

## remove everything in terms of memory full
https://qiita.com/hshimo/items/4c79cbef12ccea6d5b20
docker rm -v $(docker ps -qa)  
docker rmi -f $(docker images -q)


# tensorflow official
- https://www.tensorflow.org/install/docker
- docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
- docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu bash
