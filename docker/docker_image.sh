#!/bin/bash

img_name=tf-prode

# Check if the image already exists -------------------------------------------
if [[ "$(docker images -q $img_name:latest 2> /dev/null)" != "" ]]; then
        img=`docker images -q $img_name:latest`
	docker image rm -f $img
fi

# Create dockerfile -----------------------------------------------------------
cat << EOF > ../Dockerfile

FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="zanardi3@illinois.edu"

WORKDIR /app/prode

COPY ./ ./

RUN python3 setup.py bdist_wheel && pip3 install dist/*.whl

EOF

# Build the new image ---------------------------------------------------------
docker build \
	-t ${img_name} \
	-f ../Dockerfile \
	--cgroup-parent 'prode' \
	./

# Remove the Dockerfile -------------------------------------------------------
rm -rf ../Dockerfile