import numpy as np
import csv as csv

data = []
modified_data = []

#Import our data
#The 'r' parameter is for read and the b parameter is for 'binary'
with open('/home/rob/data/titanic_data/train.csv', 'rb') as csv_file_object:
    reader = csv.reader(csv_file_object)
    header = csv_file_object.next()
    for row in reader:
        data.append(row)

data = np.array(data)

#Calculate and print some summary stats
number_passengers = np.size(data[0::,0].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

print "---- Summary Stats ----"
print "Number of passengers in training set = %s" % number_passengers
print "Number of survivors in training set = %s" % number_survived
print "Survival rate in training set = %s" % proportion_survivors

#Calculate and print some gender-based stats
women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] == "male"
unknown_gender_stats = data[0::,4] == ""

women_onboard_idx = data[women_only_stats,0].astype(np.float)
men_onboard_idx = data[men_only_stats,0].astype(np.float)

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

#---- Age Imputation ----
#Determine which passengers have no age listed in the input file.
#  Then, calculate the average age of the passengers with an age listed.
#  Note that passengers missing age should NOT be included in the divisor
#  of the average age calculation.

passengers_missing_age = data[0::,5] == ""
count_missing_age = np.size(data[passengers_missing_age,0].astype(np.float))
sum_passenger_age = np.sum(data[- passengers_missing_age,5].astype(np.float))
average_age = sum_passenger_age / (number_passengers - count_missing_age)

print "Average (mean) age for passengers with age listed = %s" % average_age

#For passengers with no age listed, make them the average age of the rest
#   of the passengers.
data[passengers_missing_age,5] = average_age

#Remove some columns that we do not believe will be useful in our modeling
#   and then save a modified version of the file. To select different
#   columns for your model, simply change the column indices in the
#   next line.
indices_of_columns_we_want = [0, 1, 2, 4, 5, 6, 7, 9, 11]
modified_data = data[0::,indices_of_columns_we_want]

#use csv library to save new data out to new csv file
#  Subtle note: Adding the header row back in to the output file
#    is surprisingly challenging...

output_file_name = "modified_train.csv"

with open(output_file_name, 'wb') as csv_writer_object:
    writer = csv.writer(csv_writer_object)
    #writer.writerow([header])
    writer.writerows(modified_data)
    print "Done saving modified data as %s" % output_file_name
