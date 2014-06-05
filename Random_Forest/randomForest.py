# SPAM FILTER using Spambase public data set
#  Imports data from csv files
#  Plots histograms of each feature
#  Separates data into training set and test set for crossvalidation
#  Applies random forest algorithm to data set
#  Creates ROC curve for model

import matplotlib
# matplotlib.use("AGG") # enable for use on AWS node
import csv
import numpy as np
import pylab as pl

from matplotlib import pyplot as plt
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import roc_curve, auc
import prettytable as pt

# IMPORT DATA
#  open data set was downloaded into spambase folder in the hw3 directory
#  spambase data set is available at https://archive.ics.uci.edu/ml/datasets/Spambase and is saved in hw3 directory
#  spambase.data and spambase.names were renamed as spambase.csv and spambase_names.csv to facilitate the importing of data

temp_data = []

# import feature and classifications values for data set
with open("./spambase/spambase.csv", 'rb') as csv_file_object:
    reader = csv.reader(csv_file_object)
    for row in reader:
        temp_data.append(row)

temp_data = np.array(temp_data)

# separate classifcations from features in data set
spam_class = np.float_(temp_data[:,-1])
features = np.float_(temp_data[:,:-1])

# import data file with feature names
temp_data = [] 
text_file = open("./spambase/spambase.names", "r")
temp_data = text_file.readlines()
text_file.close()

# remove first 33 lines of comments in feature name file 
temp_data = temp_data[33:]

# separate string into feature name and feature type
feature_names = []

for line in temp_data:
    feature_names.append([n for n in line.split(':')])

# remove feature type from list of lists so only feature names remain
feature_names = tuple(x[0] for x in feature_names)

# segment features into separate data sets for plotting histograms for spam and non-spam 
spam = spam_class == 1
features_spam = features[spam,:].astype(np.float)
features_notspam = features[-spam,:].astype(np.float)

# plot histogram for each feature
num_bins = 20

print "\n****************************\n"
print "Plots of histograms for each feature are being created and saved in spambase folder..."
print "\n****************************\n"

# loop for creating histograms for all 57 features
for i,j in list(enumerate(feature_names)):
    #if i>2000: # FOR DEV ONLY - used to avoid making graphs during development
        # calculate max and min for ranges used in histograms
        x_max = max(features[:,i])
        x_min = min(features[:,i])
        
        # plot histogram for spam        
        plt.figure(i)
        plt.autoscale(enable=True,axis='both',tight=None)
        plt.hist(features_spam[:,i],
                 range=(0.0,x_max),
                 bins=num_bins,
                 facecolor='red',
                 label='spam',
                 alpha=.5)
        
        # plot histogram for notspam
        plt.hist(features_notspam[:,i],
                 range=(0.0,x_max),
                 bins=num_bins,
                 facecolor='blue',
                 label='not spam',
                 alpha=.5)
        
        # add labels and legend
        plt.xlabel('feature values for %s' % j)
        plt.ylabel('Count of Emails')
        plt.legend(loc="upper right")
        
        # display and/or save graphs (note: options may be commented out based on preference)
        #plt.show() # display graph
        plt.savefig('./spambase/hw3_histogram_for_feature %s .png' % j) # save figure 
                                                                
print "\n****************************\n"

# method for creating indices used to separate data sets for cross validation
def cross_val_folds(k_folds,x_train,Y_train):
    # derive a set of random training and testing indices
    k_fold_indices = KFold(len(x_train), n_folds=k_folds, indices=True, shuffle=True, random_state=0)

    return k_fold_indices
    
# method for     
def Random_Forest(x_train, Y_train,n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, 
                  random_state=0, min_density=None):
    clf = RandomForestClassifier()
    clf.fit(x_train,Y_train)
    
    return clf

# create indices for separating data into training and testing sets
folds = 2
kfold_indices = cross_val_folds(folds,features,spam_class)

# for each training and testing slice, run the classifier and build ROC curve
fold_counter = 0
pl.clf()
    
print "Plots of ROC curves are being created and saved in spambase folder..."
print "\n****************************\n"

for train_slice, test_slice in kfold_indices:
    model = Random_Forest(features[train_slice],spam_class[train_slice])
    test_predict = model.predict_proba(features[test_slice])
    score = model.score(features[test_slice],spam_class[test_slice])
    fold_counter +=1
    
    fpr, tpr, threshold = roc_curve(spam_class[test_slice],test_predict[:,1],pos_label=None)
    roc_auc = auc(fpr, tpr)

    print "For crossfold validation %d the score is %f" % (fold_counter, score)
     
    # Plot ROC curve
    pl.figure(fold_counter)
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver Operator Curve')
    pl.legend(loc="lower right")
    # display and/or save graphs (note: options may be commented out based on preference)
    #pl.show()
    pl.savefig('./spambase/hw3_ROC_curve %d .png' % fold_counter)
    
    print("Area under the ROC curve is %f" % roc_auc)
    
    #create table with thresholds, false positive rate and true positive rates
    x = pt.PrettyTable(["Threshold","False Positive Rate", "True Positive Rate"])
    x.align["Threshold"] = "l" # Left align Threshold
    x.padding_width = 1 # One space between column edges and contents (default)
    for i in range(len(fpr)):
        x.add_row([round(threshold[i],2),round(fpr[i],2),round(tpr[i],2)]) # add data rows and round to 2 digits
    print (x)
  
    print "\n****************************\n"




