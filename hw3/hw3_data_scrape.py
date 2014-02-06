import csv
import numpy as np

# import dataset from csv file and convert into numpy array
data = []

with open('./spambase/spambase.csv', 'rb') as csv_file_object:
    reader = csv.reader(csv_file_object)
    for row in reader:
        data.append(row)

data = np.array(data)

# separate classifcations from features in data set
spam_class = data[:,-1]
features = data[:,:-1]

# import feature names
feature_names_raw = [] #np.zeros(shape(57,2))

text_file = open("./spambase/spambase.names", "r")
data = text_file.readlines()

# remove first 33 lines of comments in file 
data = data[33:]
#data = np.array(data)

# remove extra spaces and separate string into feature name and feature type
for line in data:
    feature_names_raw.append([n for n in line.split(':')])
text_file.close()

#feature_names = np.array(feature_names)
# remove feature type in second column of array
#feature_names = feature_names[:,:1]
feature_names = tuple(x[0] for x in feature_names_raw)
print type(feature_names)
print len(feature_names)

#print feature_names[1]
#print feature_names[-1]

#print type(feature_names)
#print np.size(feature_names)