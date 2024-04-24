#!/bin/bash

docker build -t whisper-asr-webservice .
docker stop $(docker ps -a -q)
docker run -d -p 9001:9000 -v ~/.cache/whisper:/root/.cache/whisper -e ASR_MODEL=tiny -e ASR_ENGINE=faster_whisper whisper-asr-webservice
