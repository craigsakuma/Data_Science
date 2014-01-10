
from hw1 import load_iris_data, cross_validate, knn, naive_bayes
import argparse

# argparse that allows users to select  either KNN or NaiveBayes algorithms only
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-k","--KNN_only",action="store_true")
group.add_argument("-nb","--NaiveBayes_only",action="store_true")
args = parser.parse_args()

(XX, yy, y) = load_iris_data()

if args.KNN_only:
    classifiers_to_cv = [("kNN",knn)]

elif args.NaiveBayes_only:
    classifiers_to_cv = [("Naive Bayes",naive_bayes)]

else:
    classifiers_to_cv = [("kNN",knn),("Naive Bayes",naive_bayes)]

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
