import json
import numpy as np
import re
import operator
import random
import math
from matplotlib import pyplot as plt

funny = {}
useful = {}
cool = {}
positive = {}
funny_word = 0
useful_word = 0
cool_word = 0
positive_word = 0
total_word = 0
data = []
label_len = [0, 0, 0, 0]
funny_loss = []
useful_loss = []
cool_loss = []
positive_loss = []
train_bag = {}
default_funny_loss = []
default_useful_loss = []
default_cool_loss = []
default_positive_loss = []

"""
Function for load and filter data from json file
"""
def load(path):
    temp = []
    print('start parsing data....')
    try:
        f = open(path, 'r')
    except:
        print('Error! File does not exist!')
        return

    for line in f:
        row = json.loads(line)
        vote = row['votes']
        num = vote['funny'] + vote['useful'] + vote['cool']
        # Check vote number to decide filter the data of not
        if num >= 3  and num <=10: 
            temp.append(row)
    temp = np.array(temp)
    np.random.shuffle(temp)
    temp = temp[:16000]
    return temp

def preprocessing(sentence):
    letters = re.sub('[^a-zA-Z]', ' ', sentence)
    words_list = letters.lower().split()

    return words_list

def get_sentence(single_data):
    sentence = ""
    sentence = single_data['text']
    return sentence

def make_label(bag, some_data):
    global funny, useful, cool, positive
    global funny_word, useful_word, cool_word, positive_word, total_word
    global train_bag, label_len
    #Initialize
    for i in range(4):
        label_len[i] = 0

    funny = {}
    useful = {}
    cool = {}
    positive = {}
    funny_word = 0
    useful_word = 0
    cool_word = 0
    positive_word = 0
    total_word = 0
    train_bag = {}

    for row in some_data:
        vote = row['votes']
        text = row['text']
        stars = row['stars']
        text = preprocessing(text)
        # Calculate number of labels in training review
        if vote['funny'] > 0:
            label_len[0] += 1
        if vote['useful'] > 0:
            label_len[1] += 1
        if vote['cool'] > 0:
            label_len[2] += 1
        if stars > 3.5:
            label_len[3] += 1

        # Make bag of words for four class labels
        for word in bag:
            if word in text:
                if vote['funny'] > 0:
                    funny_word += 1
                    if word in funny:
                        funny[word] += 1
                    else:
                        funny[word] = 1
                if vote['useful'] > 0:
                    useful_word += 1
                    if word in useful:
                        useful[word] += 1
                    else:
                        useful[word] = 1
                if vote['cool'] > 0:
                    cool_word += 1
                    if word in cool:
                        cool[word] += 1
                    else:
                        cool[word] = 1
                if stars > 3.5:
                    positive_word += 1
                    if word in positive:
                        positive[word] += 1
                    else:
                        positive[word] = 1
                if word in train_bag:
                    train_bag[word] += 1
                else:
                    train_bag[word] = 1
                total_word += 1


def NBC(num, bag, test):
    global train_bag
    global data
    global total_word, label_len, funny_word, useful_word, cool_word, positive_word
    global funny, useful, cool, positive
    global funny_loss, useful_loss, cool_loss, positive_loss
    global default_funny_loss, default_useful_loss, default_cool_loss, default_positive_loss
    print('start training.... (size: {})'.format(num))
    
    funny_prob = {}
    not_funny_prob = {}
    useful_prob = {}
    not_useful_prob = {}
    cool_prob = {}
    not_cool_prob = {}
    positive_prob = {}
    not_positive_prob = {}

    tmp = data[0:num]
    # 1.(c) Convert each review of training review into bag of words and four class labels
    make_label(bag, tmp)
    isgood = 0
    # If selected data set is not approtiate, select again randomly
    while True:
        for i in label_len:
            isgood = 1
            if i == 0 or i == num:
                np.random.shuffle(data)
                tmp = data[0:num]
                make_label(bag, tmp)
                isgood = 0
                break
        if isgood == 1:
            break

    val_num = len(train_bag)
    not_funny_word = total_word - funny_word
    not_useful_word = total_word - useful_word
    not_cool_word = total_word - cool_word
    not_positive_word = total_word - positive_word 

    '''
    #For calculate base line default error
    
    if label_len[0] >= num/2:
        d_funny = True
    else:
        d_funny = False
    if label_len[1] >= num/2:
        d_useful = True
    else:
        d_useful = False
    if label_len[2] >= num/2:
        d_cool = True
    else:
        d_cool = False
    if label_len[3] >= num/2:
        d_positive = True
    else:
        d_positive = False
    '''

    # 2.(c) Code for learn the NBC model
    # Calculate conditional probablity for each bag of words and four class labels
    for word in train_bag:
        if word in funny:
            val = funny[word]
        else:
            val = 0
        funny_prob[word] = float((val+1)/(val_num+funny_word))
        val = train_bag[word] - val
        not_funny_prob[word] = float((val+1)/(not_funny_word+ val_num))
        if word in useful:
            val = useful[word]
        else:
            val = 0
        useful_prob[word] = float((val+1)/(val_num+useful_word))
        val = train_bag[word] - val
        not_useful_prob[word] = float((val+1)/(not_useful_word + val_num))
        if word in cool:
            val = cool[word]
        else:
            val = 0
        cool_prob[word] = float((val+1)/(val_num+cool_word))
        val = train_bag[word] - val
        not_cool_prob[word] = float((val+1)/(not_cool_word + val_num))
        if word in positive:
            val = positive[word]
        else:
            val = 0
        positive_prob[word] = float((val+1)/(val_num+positive_word))
        val = train_bag[word] - val
        not_positive_prob[word] = float((val+1)/(not_positive_word + val_num))
    
    # 2.(d) Read new data(test data) and predict the result
    for i in range(4):
        if i == 0:
            print('Test isFunny')
        elif i == 1:
            print('Test isUseful')
        elif i == 2:
            print('Test isCool')
        else:
            print('Test isPositive')

        cnt = 0
        #d_cnt = 0
        test_num = 0
        for line in test:
            # Probability for log10 P(Y = y)
            true_prob = math.log10(label_len[i]/num)
            false_prob = math.log10((num-label_len[i])/num)
            text = line['text']
            vote = line['votes']
            stars = line['stars']
            test_num += 1
            text = preprocessing(text)
            # Parse word from review
            if i == 0:
                if vote['funny'] > 0:
                    result = True
                else:
                    result = False
            elif i == 1:
                if vote['useful'] > 0:
                    result = True
                else:
                    result = False
            elif i == 2:
                if vote['cool'] > 0:
                    result = True
                else:
                    result = False
            else:
                if stars > 3.5:
                    result = True
                else:
                    result = False
            test_word = {}
            for word in text:
                if word in test_word:
                    test_word[word] += 1
                else:
                    test_word[word] = 1
            
            # Add conditional probability for each word in test data
            for word in test_word:         
                often = 1
                if i == 0:
                    if word in train_bag:
                        true_prob += math.log10(often*funny_prob[word])
                        false_prob += math.log10(often*not_funny_prob[word])
                elif i == 1:
                    if word in train_bag:
                        true_prob += math.log10(often*useful_prob[word])
                        false_prob += math.log10(often*not_useful_prob[word])
                elif i == 2:
                    if word in train_bag:
                        true_prob += math.log10(often*cool_prob[word])
                        false_prob += math.log10(often*not_cool_prob[word])
                else:
                    if word in train_bag:
                        true_prob += math.log10(often*positive_prob[word])
                        false_prob += math.log10(often*not_positive_prob[word])
            
            # 3.(b) Calculate Zero-One Loss
            if true_prob >= false_prob:
                predict = True
            else:
                predict = False
            if predict != result:
                cnt += 1
            '''
            if i == 0 and d_funny != result:
                d_cnt += 1
            if i == 1 and d_useful != result:
                d_cnt += 1
            if i == 2 and d_cool != result:
                d_cnt += 1
            if i == 3 and d_positive != result:
                d_cnt += 1
            '''
        print('Correct prediction: {}'.format(test_num - cnt))
        loss = float(cnt/test_num)
        #d_loss = float(d_cnt/test_num)
        print('Loss for training: {}'.format(loss))
        if i == 0:
            funny_loss.append(loss)
            #default_funny_loss.append(d_loss)
        elif i == 1:
            useful_loss.append(loss)
            #default_useful_loss.append(d_loss)
        elif i == 2:
            cool_loss.append(loss)
            #default_cool_loss.append(d_loss)
        else:
            positive_loss.append(loss)
            #default_positive_loss.append(d_loss)
    print('\n')
        

if __name__ == "__main__":
    # 1.(a) Filter the reviews data to based on number of votes.
    data = load('yelp_academic_dataset_review.json')
    bag_all = {}
    bag = {}
    # 1.(b) Compute 3000 most frequently occuring words
    # First compute all word's frequency in filtered data
    for single_data in data:
        sentence = get_sentence(single_data)
        words_list = preprocessing(sentence)
        for word in words_list:
            if word in bag_all:
                bag_all[word] += 1
            else:
                bag_all[word] = 1
    # Use list to sort word's frequency to decide word frequency cutline
    count = []
    for word in bag_all:
        count.append(bag_all[word])
    count.sort(reverse=True)
    flag = count[2999]
    cnt = 0
    # Add 3000 most frequently occured words in to bag
    for word in bag_all:
        if bag_all[word] > flag:
            bag[word] = bag_all[word]
            cnt += 1
    for word in bag_all:
        if bag_all[word] == flag:
            bag[word] = bag_all[word]
            cnt += 1
        if cnt == 3000:
            break
    print('Finished parsing data')
    total = len(data)
    # 3.(a) Create training set and test set randomly
    # Randomly shuffle the data
    np.random.shuffle(data)
    # Get test data
    test = data[0:3200]
    data = data[3200:]
    training_array = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]

    # Start training & test
    for i in range(9):
        # Randimly select training data
        np.random.shuffle(data)
        NBC(training_array[i], bag, test)

    # Draw Loss graph of test data for each training set
    sample = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.plot(sample, funny_loss)
    plt.plot(sample, useful_loss)
    plt.plot(sample, cool_loss)
    plt.plot(sample, positive_loss)
    plt.xlabel('Train number')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.legend(['isFunny', 'isUseful', 'isCool', 'isPositive'])
    plt.show()
    '''
    plt.plot(sample, default_funny_loss)
    plt.plot(sample, default_useful_loss)
    plt.plot(sample, default_cool_loss)
    plt.plot(sample, default_positive_loss)
    plt.xlabel('Train number')
    plt.ylabel('Default Loss')
    plt.title('Default Loss Graph')
    plt.legend(['isFunny', 'isUseful', 'isCool', 'isPositive'])
    plt.show()
    '''
    