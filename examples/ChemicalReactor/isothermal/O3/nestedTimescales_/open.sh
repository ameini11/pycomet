#!/bin/bash -i

# Open PDF images
# -----------------------------------------------------------------------------

# Default arguments
DIR='./'
FILE='TS2'

# Parsing arguments
while [[ $# -gt 0 ]]; do
	key="$1"

	case $key in
	    -d|--dir)
			DIR="$2"
			shift # past argument
			shift # past value
			;;
	    -f|--file)
			FILE="$2"
			shift # past argument
			shift # past value
			;;
	    *)
			echo "Unknown option '${1}'"
            exit 1
			;;
	esac
done

for d in $DIR/*;
do
    echo "Showing '$d'"
    evince $d/$FILE'.pdf'
done
