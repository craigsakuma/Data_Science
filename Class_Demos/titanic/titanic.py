import numpy as np
import csv as csv

#Import our data
csv_file_object = csv.reader(open('/home/rob/data/titanic_data/train.csv', 'rb'))
header = csv_file_object.next()
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

#Calculate some summary stats
number_passengers = np.size(data[0::,0].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

print "---- Summary Stats ----"
print "Number of passengers in training set = %s" % number_passengers
print "Number of survivors in training set = %s" % number_survived
print "Survival rate in training set = %s" % proportion_survivors

#Calculate some gender-based stats
women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] == "male"
unknown_gender_stats = data[0::,4] == ""

passengers_missing_age = data[0::,5] == ""
count_missing_age = np.size(data[passengers_missing_age,0].astype(np.float))
sum_passenger_age = np.sum(data[- passengers_missing_age,5].astype(np.float))
average_age = sum_passenger_age / (number_passengers - count_missing_age)
print average_age

data[passengers_missing_age,5] = average_age

#use csv library to save new data out to new csv file

print passengers_missing_age

women_onboard_idx = data[women_only_stats,0].astype(np.float)
men_onboard_idx = data[men_only_stats,0].astype(np.float)

#print women_onboard_idx

number_women_onboard = np.size(data[women_only_stats,0].astype(np.float))
number_men_onboard = np.size(data[men_only_stats,0].astype(np.float))
number_unknown_gender_onboard = np.size(data[unknown_gender_stats,0].astype(np.float))

print "---- Gender-Based Stats ----"
print "number of women onboard = %s" % number_women_onboard
print "number of men onboard = %s" % number_men_onboard
print "number of unknown gender onboard = %s" % number_unknown_gender_onboard

number_women_survivors = np.sum(data[women_only_stats,1].astype(np.float))
number_men_survivors = np.sum(data[men_only_stats,1].astype(np.float))

print "number of women survivors = %s" % number_women_survivors
print "number of men survivors = %s" % number_men_survivors

proportion_women_survived = number_women_survivors / number_women_onboard
proportion_men_survived = number_men_survivors / number_men_onboard

print "Proportion of women who survived = %s" % proportion_women_survived
print "Proportion of men who survived = %s" % proportion_men_survived
