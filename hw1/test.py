
from hw1 import load_iris_data, cross_validate, knn, naive_bayes

(XX, yy, y) = load_iris_data()

classifiers_to_cv = [("kNN",knn),("Naive Bayes",naive_bayes)]

for (c_label, classifier) in classifiers_to_cv:

    print
    print "---> %s <---" % c_label
    
    best_k = 0
    best_cv_a = 0

    for k_f in [2,3,5,10,15,30,50,75,150]:
         cv_a = cross_validate(XX, yy, classifier, k_fold=k_f)
         if cv_a > best_cv_a:
             best_cv_a = cv_a
             best_k= k_f
         print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)
    print "Highest accuracy: fold <<%s>> ::<%s>>"% (best_k, best_cv_a)
