#
# Tutorial 9, Question 2
#

import numpy as np
import csv

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    

csvfile = open('international-airline-passengers.csv')
csvreader = csv.reader(csvfile)

months = [row[0] for row in csvreader if len(row) > 0]
print('{}'.format(months))

csvfile.close()


csvfile = open('international-airline-passengers.csv')
csvreader = csv.reader(csvfile)

demand = [float(row[1]) for row in csvreader if len(row) > 0]
print('{}'.format(demand))

csvfile.close()

for i in range(10):
    print('{} : {}'.format(months[i], demand[i]))
