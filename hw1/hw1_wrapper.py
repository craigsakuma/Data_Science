# Wrapper that imports iris data set and scores a classifier (KNN and Naive Bayes) using a variety
# of cross validation scenarios to determine optimal sample size for training the classifier which avoids over fitting 

from hw1_functions import load_iris_data, cross_validate, knn, naive_bayes
import argparse

# argparse that allows users to select only KNN classifier or only NaiveBayes algorithms.  Default will run both classifiers
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-k","--KNN_only",action="store_true")
group.add_argument("-nb","--NaiveBayes_only",action="store_true")
args = parser.parse_args()

# load iris data set
(XX, yy, y) = load_iris_data()

# if argparse is used to select KNN_only
if args.KNN_only:
    classifiers_to_cv = [("kNN",knn)]

# if argparse is used to select NaiveBayes_only
elif args.NaiveBayes_only:
    classifiers_to_cv = [("Naive Bayes",naive_bayes)]

# default selection which includes both classifiers if argparse isn't used
else:
    classifiers_to_cv = [("kNN",knn),("Naive Bayes",naive_bayes)]

# for each classifier
# train and score the model using n folds cross validation
# for each value of n calculate and print the average of all the scores from the cross validation
# print the highest accuracy n fold with score  
for (c_label, classifier) in classifiers_to_cv:

    print "\n---> %s <---" % c_label
    
    best_k = 0
    best_cv_a = 0

    for k_f in [2,3,5,10,15,30,50,75,150]:
         cv_a = cross_validate(XX, yy, classifier, k_fold=k_f)
         if cv_a > best_cv_a:
             best_cv_a = cv_a
             best_k= k_f
         print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)
    print "Highest accuracy: fold <<%s>> ::<%s>>"% (best_k, best_cv_a)
print "\n"
