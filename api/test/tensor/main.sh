#!/bin/bash
set -e

source ./../env.sh

make

./test

make clean
