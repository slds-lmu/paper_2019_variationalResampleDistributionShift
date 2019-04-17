# list all running containers
docker ps -a
## CONTAINER ID IMAGE COMMAND CREATED STATUS...

# re-entering a running docker
docker exec -it [container-id] bash

# pull and run a container
nvidia-docker run -it tensorflow/tensorflow:1.4.1-gpu
nvidia-docker run -it tensorflow/tensorflow:1.13.1 
