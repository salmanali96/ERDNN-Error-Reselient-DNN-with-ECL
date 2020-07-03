#!/usr/bin/env bash

echo "Make Error LookUp Table"
for err in 0.1 0.2 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.001 0.0001 0.00001
do
    echo "Error rate: $err"
    python3 LUT.py -e $err
done