#!/bin/bash

# install python requirements
echo "installing requirements"
pip install -r requirements.txt

echo "-- SemEval2010 data --"
mkdir -p data/SemEval2010
cd data/SemEval2010

echo "Downloading zip"
wget wget -q --show-progress https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/SemEval2010.zip | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'

echo "Extracting zip"
unzip SemEval2010.zip | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'

cd ..
cd ..

echo "-- Inspec data --"
mkdir -p data/Inspec
cd data/Inspec

echo "Downloading zip"
wget wget -q --show-progress https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Inspec.zip | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'

echo "Extracting zip"
unzip Inspec.zip | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'

cd ..
cd ..

echo "-- 500N-KPCrowd data --"
mkdir -p data/500N-KPCrowd-v1.1
cd data/500N-KPCrowd-v1.1

echo "Downloading zip"
wget wget -q --show-progress https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/500N-KPCrowd-v1.1.zip 

echo "Extracting zip"
unzip -o 500N-KPCrowd-v1.1.zip | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'

python3 -m nltk.downloader stopwords
python3 -m nltk.downloader punkt
python3 -m nltk.downloader universal_tagset
python3 -m spacy download en_core_web_sm # download the english model

echo "Setup was successfull!"