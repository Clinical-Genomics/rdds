#!/bin/bash

set -e

CASES_DIR=/home/tor.bjorgen/repos/cg/binarycross-w-augmentation/tmp/test_cases_predictions_20250318-100622-be25519

for dir in `ssh hasta find $CASES_DIR -type d`;
do
	if [ $dir == $CASES_DIR ]; then
		continue
	fi
	case_name=`basename $dir`
	mkdir -p $case_name
	scp -r hasta:$CASES_DIR/$case_name/*.png $case_name
done
