#!/usr/bin/env bash

for file in /mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/domain_randomization/potato5050/*
do
    opencv_createsamples -img $file -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950
done