from hw2 import load_iris_data, cross_validate, log_reg
import numpy as np

# load data set
(XX, yy, y) = load_iris_data()

# create variable with test values for C
c_values = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
c_label = "Logistic Regression"
best_c = 0
best_score_c = 0

# user input for number of folds for cross validation
k_f = int(raw_input("Enter number of folds for cross validation:"))


print "\n---> %s <---" % c_label
    
for c in c_values:
    c_score = cross_validate(XX, yy, log_reg, k_f, c)
    if c_score > best_score_c:
        best_score_c = c_score
        best_c = c
    print "C-Value <<%s>> :: acc <<%s>>" % (c, c_score)
print "Highest accuracy: C-Value <<%s>> ::<%s>>"% (best_c, best_score_c)
print "\n"
