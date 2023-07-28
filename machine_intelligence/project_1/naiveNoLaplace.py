#!/usr/bin/python3
# A Naieve bayes classifier for Cyber Analytics and Machine Learning
# Author: Manish Sharma

# imports
import math
import re

# Define variables
training = .7
spamprobability = .1 # as stated in the assignment
hamprobability = .9


def remove_special_characters(line):
    return (re.sub('[^A-Za-z0-9 ]+', '', line))


def import_data(filename):
    output = []
    for i in open(filename, 'r'):
        split_data = i.split("\t")
        classification_tmp = (0 if split_data[0] == 'ham' else 1)
        data_tmp = (split_data[1])
        output.append([classification_tmp, data_tmp])
    return output


def process(line):
    data = remove_special_characters(line.lower()).split(" ")
    # surprisingly, this is the best way to remove all instances of '' from the data
    return data


def generate_probability_table(ham_wordlist, spam_wordlist):
    ##expect a ham wordlist and spam wordlist with frequencies for each
    # this method determines the frequency based on ham vs spam and returns a table
    freq_table = {}
    total_ham = len(ham_wordlist)
    total_spam = len(spam_wordlist)
    for i in ham_wordlist.keys():
        freq_table[i] = [ham_wordlist[i], 1]
    for i in spam_wordlist.keys():
        if (freq_table.get(i) != None):
            count = freq_table[i]
            count[1] = spam_wordlist[i]
        else:
            freq_table[i] = [1, spam_wordlist[i]]
    final_table = {}
    for i in freq_table.keys():
        final_table[i] = [(freq_table[i][0])/ (total_ham * len(freq_table)), (freq_table[i][1])/ (total_spam * len(freq_table))]
    return final_table


def train(data):
    # first thing i'm doing here is going through and finding probabilities
    # this method expects an import_data processed 2d array that contains a 0 or 1 for ham and spam, and a string of text to process.
    ham_wordlist = dict()
    spam_wordlist = dict()
    for i in data:
        processed_words = process(i[1])
        for word in processed_words:
            if (i[0] == 0):
                if (ham_wordlist.get(word) == None):
                    ham_wordlist[word] = 2
                else:
                    ham_wordlist[word] = ham_wordlist[word] + 1
            elif (i[0] == 1):
                if (spam_wordlist.get(word) == None):
                    spam_wordlist[word] = 2
                else:
                    spam_wordlist[word] = spam_wordlist[word] + 1
    table = generate_probability_table(ham_wordlist, spam_wordlist)
    return table


def calculate(data, trained_table):
    spamtotal = spamprobability
    hamtotal = hamprobability

    processed_words = process(data[1])
    for word in processed_words:
        if (trained_table.get(word) != None):
            if(trained_table[word][1] != 0):
                spamtotal *= trained_table[word][1]
            if (trained_table[word][0] != 0):
                hamtotal *= trained_table[word][0]

    if(spamtotal >= hamtotal):
        return 1
    else:
        return 0


def calculate_ham(data, trained_table, pham):
    hamtotal = []

    for i in data:
        total = pham
        processed_words = process(i[1])
        for word in processed_words:
            if (trained_table.get(word) == None):
                total += 0
            else:
                if (trained_table[word][1] != 0):
                    total *= trained_table[word][0]
        hamtotal.append(total)
    return hamtotal


def main():
    filedata = import_data('SMSSpamCollection')
    traindata = filedata[:math.floor(training * len(filedata))]
    testdata = filedata[len(filedata) - math.floor(training * len(filedata)):]  # was missing a :
    table = train(traindata)

    tpcount = 0 # Correctly labeled spam
    tncount = 0 # Correctly labeled ham
    fpcount = 0 # Ham labeled as spam
    fncount = 0 # Spam labeled as ham

    answers = []
    for i in testdata:
        answers.append(calculate(i, table))

    spamcount = 0
    hamcount = 0
    counter = 0

    for i in testdata:
        if i[0] == 1 and answers[counter] == 1:
            tpcount += 1
        elif i[0] == 1 and answers[counter] == 0:
            fpcount += 1
        elif i[0] == 0 and answers[counter] == 0:
            tncount += 1
        else:
            fncount += 1
        counter += 1

    accuracy = (tpcount + tncount) / (tpcount + fpcount + tncount + fncount) * 100
    precision = tpcount / (tpcount + fpcount) * 100
    recall = tpcount / (tpcount + fncount) * 100

    print('TP: ', tpcount)
    print('FP: ', fpcount)
    print('TN: ', tncount)
    print('FN: ', fncount)
    print('Accuracy: ', accuracy, '%')
    print('Precision: ', precision, '%')
    print('Recall: ', recall, '%')

    test1 = [0, "dude! dude! look!"]
    test2 = [1, "winner babe! click for prize"]
    print("Test SMS 1:", calculate(test1,table))
    print("Test SMS 2:", calculate(test2, table))



main()