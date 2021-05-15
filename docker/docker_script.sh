#!/bin/bash

img_name=tf-prode
run_num=0

# Define the name of the new image --------------------------------------------
name=${img_name}_run${run_num}

# Check if the image already exists -------------------------------------------
if [[ "$(docker images -q ${name}:latest 2> /dev/null)" != "" ]]; then
	img=`docker images -q ${name}:latest`
	docker image rm -f $img
fi

# Create dockerfile -----------------------------------------------------------
cat << EOF > Dockerfile

FROM ${img_name}:latest

LABEL maintainer="zanardi3@illinois.edu"

WORKDIR /run${run_num}

COPY ./ ./

CMD ["main.py"]

ENTRYPOINT ["python3"]

EOF

# Copy all the needed files/folders inside the local run directory ------------
cp -r $(pwd)/../../database .
cp -r $(pwd)/../../utils .

# Build the new image ---------------------------------------------------------
docker build \
	-t ${name} \
	-f Dockerfile \
	--cgroup-parent 'prode' \
	./

# Run the new image -----------------------------------------------------------
docker run \
	--gpus all \
	--name run${run_num} \
	-v "$(pwd)":/run${run_num} \
	-it \
	--rm \
	${name}:latest

# Remove the image ------------------------------------------------------------
docker image rm \
	-f \
	${name}:latest
