#!/bin/bash

# =========================================================================== #
# Build `prode` image in docker
# =========================================================================== #
# Instructions
# >> run this script with docker root privileges
# >> define the following variables:
parent=tensorflow/tensorflow	# Parent image on top of which the new one is created
tag=latest-gpu					# Tag of the parent image
# =========================================================================== #

img_name=tf-prode

# Check if the image already exists -------------------------------------------
if [[ "$(docker images -q $img_name:latest 2> /dev/null)" != "" ]]; then
        img=`docker images -q $img_name:latest`
	docker image rm -f $img
fi

# Create the Dockerfile -------------------------------------------------------
cat << EOF > ../Dockerfile

FROM ${parent}:${tag}

LABEL maintainer="zanardi3@illinois.edu"

WORKDIR /app/prode

COPY ./ ./

RUN python3 setup.py bdist_wheel && pip3 install dist/*.whl

EOF

# Build the new image ---------------------------------------------------------
docker build \
	-t ${img_name} \
	-f ../Dockerfile \
	--cgroup-parent ${parent} \
	./

# Remove the Dockerfile -------------------------------------------------------
rm -rf ../Dockerfile