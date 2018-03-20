#!/bin/bash

#
# MAKE THE DATA FOLDER
#

DST=datasets
mkdir -p $DST
cd $DST

#
# DOWNLOAD THE UKB IMAGES
#

wget https://archive.org/download/ukbench/ukbench.zip
unzip ukbench.zip
mv full ukbench

#
# CLEAN UP
#

rm ukbench.zip
rm -rf thumbnails/
rm *.html
rm find_duplicates.sh