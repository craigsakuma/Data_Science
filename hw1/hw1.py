from sklearn import datasets
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

KNN = True
NB = False 


# Questions
# How to separate data set into randomly ordered groups for training and testing (e.g., iris_data and iris_target)
# Numpy - arange and shuffle?








# Pseudo Code Notes
# 
# Load data
#     Training data
#     Data for prediction
#     Data for scoring
# 
# Choose model
#     if KNN choose neighbors
#     additional parameters for either model
#
# Cross Validation
#     get KFolds parameteres
#     test multiple KFolds      
#     visualize results
#
# Create django interface
# 






def load_iris_data():
    # loads iris dataset 

    iris = datasets.load_iris()

    return (iris.data, iris.target, iris.target_names)


def knn(X_train, y_train, k_neighbors = 3):
    # method returns a kNN object called clf with methods:
    #     score(X_test, y_test) --> to score the model using a test
    #     predict(X_classify, y_test) --> to predict a result using

    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf

def naive_bayes(X_train, y_train):
    # method returns a NB object called clf with methods

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf


def cross_validate(XX, yy, classifier, k_fold):
    # function returns generic cross validation

    # derive a set of random training and testing indices
    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)

    k_score_total = 0
    # for each training and testing slice, run the classifier and score
    for train_slice, test_slice in k_fold_indices:

        model = classifier(XX[[ train_slice ]],
                           yy[[ train_slice ]])


#        if classifier == "KNN":
#            model = knn(XX[[ train_slice ]],
#                        yy[[ train_slice ]])

#        elif classifier == "NB":
#            model = naive_bayes(XX[[ train_slice ]],
#                                yy[[ train_slice ]]) 
        
        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        # score each slice and average to determine performance of model
        k_score_total += k_score

    return k_score_total*1.0/k_fold
