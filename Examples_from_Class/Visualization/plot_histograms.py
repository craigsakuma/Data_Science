import matplotlib
matplotlib.use("AGG")
from matplotlib import pyplot as plt
import numpy as np
import csv as csv

data = []
modified_data = []
output_file_name = "output.csv"
input_file_name = "/home/rob/data/titanic_data/train.csv"
num_bins = 20

#Import our data
#The 'r' parameter is for read and the b parameter is for 'binary'
with open(input_file_name, 'rb') as csv_file_object:
    reader = csv.reader(csv_file_object)
    header = csv_file_object.next()
    for row in reader:
        data.append(row)

data = np.array(data)

passengers_missing_age = data[0::,5] == ""
data_with_ages = data[-passengers_missing_age,0:]

ages_not_missing = data[-passengers_missing_age, 5].astype(np.float)
class_labels_ages_not_missing = data[-passengers_missing_age,1]

passengers_died = data_with_ages[0:,1] == "0"
passengers_survived = data_with_ages[0:,1] == "1"
ages_died = data_with_ages[passengers_died, 5].astype(np.float)
ages_survived = data_with_ages[passengers_survived, 5].astype(np.float)

plt.figure(1)
plt.hist(ages_not_missing[0:],
         bins=num_bins,
         range=(0,100),
         facecolor='green',
         label='Class',
         alpha=.5)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.savefig('hist_all_pass.png')

plt.figure(2)
plt.hist(ages_died[0:],
         bins=num_bins,
         range=(0,100),
         facecolor='red',
         label='Class',
         alpha=.5)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.savefig('hist_died.png')

plt.figure(3)
plt.hist(ages_survived[0:],
         bins=num_bins,
         range=(0,100),
         facecolor='blue',
         label='Class',
         alpha=.5)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.savefig('hist_survived.png')

plt.figure(4)
plt.hist(ages_died[0:],
         bins=num_bins,
         range=(0,100),
         facecolor='red',
         label='Class',
         alpha=.5)

plt.hist(ages_survived[0:],
         bins=num_bins,
         range=(0,100),
         facecolor='blue',
         label='Class',
         alpha=.5)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.savefig('hist_survived_vs_died.png')

plt.figure(5)
plt.hist([ages_survived[0:],ages_died[0:]],
         bins=num_bins,
         range=(0,100),
         histtype = 'barstacked',
         #facecolor='blue',
         label='Class',
         alpha=.5)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.savefig('hist_stacked.png')

print "Done plotting histograms!"
