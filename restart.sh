#!/bin/bash

docker build -t whisper-asr-webservice .
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker run -d -p 9001:9000 -v ~/.cache/whisper:/root/.cache/whisper -e ASR_MODEL=tiny -e ASR_ENGINE=faster_whisper -e API_SECRET=test123 whisper-asr-webservice
