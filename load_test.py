import numpy as np
import pandas as pd
import string
import copy
import sys
import pprint
import re
import csv
from collections import Counter

TAGS = {'O':0,'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6,'JJ':7,'JJR':8,'JJS':9,'LS':10,'MD':11,'NN':12,\
    'NNS':13,'NNP':14,'NNPS':15,'PDT':16,'POS':17,'PRP':18,'PRP$':19,'RB':20,'RBR':21,'RBS':22,'RP':23,\
    'SYM':24,'TO':25,'UH':26,'VB':27,'VBD':28,'VBG':29,'VBN':30,'VBP':31,'VBZ':32,'WDT':33,\
    'WP':34,'WP$':35,'WRB':36, 'END':37}
TAG_IDS = {v:k for k,v in TAGS.items()}
NUM_TAGS = len(TAGS)
NUM_TAGS_SQR = NUM_TAGS**2
PRUNED_PUNCTUATION = '!""#\\\'\'()*+,--./:;<=>?[]^_``{|}~'
PUNCTUATION_TAGS = {'#':'#','\'\'':'\'\'','(':'(','{':'(',')':')','}':')',',':',','!':'.','.':'.','?':'.','-':':','--':':','...':':',':':':',';':':','`':'``','``':'``','non-``':'``'}
PERIOD_TAGS = ['.', '!', '?']

def load_data_split(sentence_file, tag_file=None):
    sentences = []
    sentence = ['O']
    sentences_tags = None
    docstart = [0]
    end_without_period = []
    index = 0    

    if (tag_file):
        sentences_tags = []
        sentence_tags = ['O']
        with open(sentence_file, 'r') as x_data, open(tag_file, 'r') as y_data:
            next(x_data)
            next(x_data)
            next(y_data)
            next(y_data)
            for line1, line2 in zip(x_data, y_data):
                word = line1.split(',',1)[1].strip()[1:-1]
                tag = line2.split(',',1)[1].strip()[1:-1]
                if word.strip() == '-DOCSTART-':  
                    if len(sentence) == 1 and sentence[-1] == 'O': 
                        docstart.append(len(sentences))
                        continue 
                    else:
                        end_without_period.append(len(sentences))
                        sentence.append("END")
                        sentence_tags.append('END')
                        sentences.append(sentence)
                        sentences_tags.append(sentence_tags)
                        sentence = ['O']
                        sentence_tags = ['O']
                        docstart.append(len(sentences))
                        
                elif word.strip() in PERIOD_TAGS:
                    sentence.append("END")
                    sentence_tags.append("END")
                    sentences.append(sentence)
                    sentences_tags.append(sentence_tags)
                    sentence = ['O']
                    sentence_tags = ['O']
                    index += 1
                else:
                    sentence.append(word)
                    sentence_tags.append(tag)
            if len(sentence) > 1 or sentence[-1] != 'O': 
                sentence.append("END")
                sentences.append(sentence)
                sentence_tags.append("END")
                sentences_tags.append(sentence_tags)
    else:
        with open(sentence_file, 'r') as x_data:
            next(x_data) # Skip first -DOCSTART-
            next(x_data)
            for line in x_data:
                word = line.split(',',1)[1].strip()[1:-1]
                docstart.append(index)
                if word.strip() == '-DOCSTART-':
                    docstart.append(index)
                    if len(sentence) == 1 and sentence[-1] == 'O': 
                        continue 
                    else:
                        sentence.append("END")
                        sentences.append(sentence)
                        sentence = ['O']
                        index += 1
                elif word.strip() in PERIOD_TAGS:
                    sentence.append("END")
                    sentences.append(sentence)
                    sentence = ['O'] 
                    index += 1                  
                else: 
                    sentence.append(word)
            sentence.append('END')
            sentences.append(sentence)

    return sentences, sentences_tags, docstart, end_without_period

def format_output_split(tags, docstart, end_without_period):
    temp = [[tag for tag in line if tag != 'O'] for line in tags]
    for i in range(len(docstart)):
        temp[docstart[i]].insert(0,'O')
    for i in range(len(temp)):
        if i in end_without_period: continue
        temp[i].insert(len(temp[i]), '.')
    tag_list = [tag for line in temp for tag in line if tag != 'END']
    return tag_list

def format_output_sentences_split(sentences, docstart, end_without_period):
    for i in range(len(docstart)):
        sentences[docstart[i]].insert(0,'-DOCSTART-')
    for i in range(len(sentences)):
        if i in end_without_period: continue
        sentences[i].insert(len(sentences[i]), '.')
    words = [word for line in sentences for word in line if word != 'END' and word != 'O']
    return words

def prune_sentences(sentences):
    """ Prune punctuations for test set """
    pruned_sentences = [[word for word in sentence if (word not in PUNCTUATION_TAGS.keys()) and not "$" in word] for sentence in sentences]
    return pruned_sentences

def deprune_formatted_output_from_sentences(original_sentences, tags):
    for i in range(len(original_sentences)):
        if original_sentences[i] in PUNCTUATION_TAGS.keys():
            tags.insert(i,PUNCTUATION_TAGS[original_sentences[i]])
        elif "$" in original_sentences[i]: 
            tags.insert(i,"$")

if __name__ == "__main__":
    # TODO: compare pruning techniques
    # figure out punctuation cleaning without tags?
    # 

    train_x,train_y,docstart,end = load_data_split("data/mini_x.csv", "data/mini_y.csv")
    print(docstart)
    #[0, 7, 13, 20, 25, 70]
    # for i in range(len(docstart)):
    #     print(train_x[docstart[i]][:5])
    print(end)
    # end = [x+1 for x in end]
    # for i in range(len(end)):
    #     print(train_x[end[i]][:5])

    x = format_output_sentences_split(train_x, docstart,end)
    y = format_output_split(train_y, docstart, end)
    
    
    deprune_formatted_output_from_sentences(x, y)
    filex = open("mini_test_x.csv", "w")
    filex.write("id,word"+"\n")
    for i in range(len(x)):
        filex.write(str(i)+",\"" + x[i] +"\"" + "\n")
    filex.close()


    