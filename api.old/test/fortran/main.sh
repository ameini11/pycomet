#!/bin/sh

set -e

. ../env.sh

make

./test

make clean
