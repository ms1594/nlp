#!/usr/bin/python3
#A Naieve bayes classifier for Cyber Analytics and Machine Learning
#Author: Manish Sharma

#imports
import math
import re

#Define variables
training = .7

def remove_special_characters(line):
    return(re.sub('[^A-Za-z0-9 ]+', '', line))

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
    #surprisingly, this is the best way to remove all instances of '' from the data
    return data

def generate_probability_table(ham_wordlist, spam_wordlist):
    ##expect a ham wordlist and spam wordlist with frequencies for each
    #this method determines the frequency based on ham vs spam and returns a table
    freq_table = {}
    total_ham = len(ham_wordlist)
    total_spam = len(spam_wordlist)
    for i in ham_wordlist.keys():
        freq_table[i] = [ham_wordlist[i], 1]
    for i in spam_wordlist.keys():
        if(freq_table.get(i) != None):
            count = freq_table[i]
            count[1] = spam_wordlist[i]
        else:
            freq_table[i] = [1, spam_wordlist[i]]
    final_table = {}
    for i in freq_table.keys():
        final_table[i] = [math.log(freq_table[i][0]/total_ham), math.log(freq_table[i][1]/total_spam)]
    return final_table
  

def train(data):
    #first thing i'm doing here is going through and finding probabilities
    #this method expects an import_data processed 2d array that contains a 0 or 1 for ham and spam, and a string of text to process.
    ham_wordlist = dict()
    spam_wordlist = dict()
    for i in data:
        processed_words = process(i[1])
        for word in processed_words:
            if(i[0] == 0):
                if(ham_wordlist.get(word) == None):
                    ham_wordlist[word] = 2
                else:
                    ham_wordlist[word] = ham_wordlist[word] + 1
            elif(i[0] == 1):
                if(spam_wordlist.get(word) == None):
                    spam_wordlist[word] = 2
                else:
                    spam_wordlist[word] = spam_wordlist[word] + 1
    table = generate_probability_table(ham_wordlist, spam_wordlist)
    return table


def main():
    filedata = import_data('SMSSpamCollection')
    traindata = filedata[:math.floor(training*len(filedata))]
    testdata = filedata[len(filedata) - math.floor(training * len(filedata))]
    table = train(traindata)
    print(len(table))

main()