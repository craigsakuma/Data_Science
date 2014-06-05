# Functions for hw2_wrapper.py
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression


def load_iris_data():
    # loads iris dataset 
    iris = datasets.load_iris()

    return (iris.data, iris.target, iris.target_names)


def log_reg(X_train, y_train,C_param):
    # method returns a logistic regression object called clf
    clf = LogisticRegression(C=C_param)
    clf.fit(X_train, y_train)

    return clf


def cross_validate(XX, yy, classifier, k_fold,c):
    # derive a set of random training and testing indices
    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)

    k_score_total = 0
    # for each training and testing slice, run the classifier and score
    for train_slice, test_slice in k_fold_indices:

        model = classifier(XX[[ train_slice ]],
                           yy[[ train_slice ]],c)

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        # score each slice and average to determine performance of model
        k_score_total += k_score

    return k_score_total*1.0/k_fold


