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
nvidia-docker run -it tensorflow/tensorflow:1.4.1-gpu  # 1.4.1 is the latest tensorflow that support cuda 8
nvidia-docker run -it tensorflow/tensorflow:1.13.1

# debug
make sure  "nvidia-smi" works
docker rmi -f <image ID> and rerun command  # does not work
