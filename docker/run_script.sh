#!/bin/bash

# =========================================================================== #
# Run python script
# =========================================================================== #
# Instructions
# >> define the following input parameters first:
parent=pycomet                  # Parent image
tag=latest                      # Tag of the parent image
run_num=0                       # Running number
cmd='pycomet train -j 4 ./'     # Command
# >> run this script with docker root privileges
# >> run this script inside a directory with the 'inputs' folder
# =========================================================================== #


# Construct the name of the new image -----------------------------------------
name=${parent}_run${run_num}

# Check if the image already exists -------------------------------------------
if [[ "$(docker images -q ${name}:${tag} 2> /dev/null)" != "" ]]; then
    img=`docker images -q ${name}:${tag}`
    docker image rm -f $img
fi

# Create the Dockerfile -------------------------------------------------------
cat << EOF > Dockerfile

FROM ${parent}:${tag}

WORKDIR /home/run${run_num}

COPY ./ ./

CMD ["${cmd}"]

EOF

# Build the new image ---------------------------------------------------------
docker build \
    -t ${name} \
    -f Dockerfile \
    --cgroup-parent ${parent} \
    ./

# Run the new image inside a container ----------------------------------------
docker run \
    --gpus all \
    --name run${run_num} \
    -v "$(pwd)":/home/run${run_num} \
    -it \
    --rm \
    ${name}:${tag}

# Remove the image ------------------------------------------------------------
docker image rm \
    -f \
    ${name}:${tag}

# Remove the Dockerfile -------------------------------------------------------
rm -rf Dockerfile
