#!/bin/bash -i

set -e

. ../env.sh

make

./test

make clean
