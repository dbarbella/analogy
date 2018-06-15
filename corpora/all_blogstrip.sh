#!/bin/bash

for file in ./blogs/*.xml
do
	python blogstrip.py $file
done

echo "Finished"
