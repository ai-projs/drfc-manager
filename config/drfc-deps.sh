#!/usr/bin/env bash

mkdir drfc-images

git clone https://github.com/aws-deepracer-community/deepracer-for-cloud.git

cd deepracer-for-cloud || exit

mv docker/* ../drfc-images

cd ..

sudo rm -r deepracer-for-cloud

