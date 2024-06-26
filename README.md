# FaceAPI

This project is mainly focused on creating a docker container for FaceAPI using several open-sourced pretrained pytorch models (purely implemented on python)

## Getting Started

These instructions will cover usage information and for the docker container 

### Prerequisities


In order to run this container you'll need docker installed.

* [Windows](https://docs.docker.com/windows/started)
* [OS X](https://docs.docker.com/mac/started/)
* [Linux](https://docs.docker.com/linux/started/)

### Usage

#### Container Parameters

To run the container

```shell
docker run -d -p 8000:8000 sethukrishna344/faceapi:latest
```

Start a shell in the container with

```shell
docker exec -it <container-name> -- /bin/bash
```

#### Environment Variables

* `SLEEP_FOR` - Waiting time for connecting to milvus db
* `MILVUS_URL` - Where does milvus db exists (Ex: http://localhost:19530)

#### Useful File Locations
* `/app` - Project Directory location within the container

* `/app/main.py` - Main python script for FastAPI server
  
* `/app/pretrained_models` - For all the pretrained models

## Built With

* Docker Engine v26.1.1
* Image architecture linux/amd64
* python libraries versions can be checked on [requirements.txt](/requirements.txt)

## Key libraries/frameworks used:
- FastAPI (For creating the API)
- Milvus (Database for face similarity search)
- facenet_pytorch python library (for face detection, recognition models)
- silent face antispoofing library (for face antispoofing detection)

