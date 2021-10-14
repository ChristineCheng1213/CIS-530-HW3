""" Contains the part of speech tagger class. """
import numpy as np
import pandas as pd
import string
import copy
import sys
import pprint
import re
import csv
from collections import Counter

# TODO:
#     - update beam search bigram
#     - split by sentence 
#     - smoothing (?)

TAGS = {'O':0,'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6,'JJ':7,'JJR':8,'JJS':9,'LS':10,'MD':11,'NN':12,\
    'NNS':13,'NNP':14,'NNPS':15,'PDT':16,'POS':17,'PRP':18,'PRP$':19,'RB':20,'RBR':21,'RBS':22,'RP':23,\
    'SYM':24,'TO':25,'UH':26,'VB':27,'VBD':28,'VBG':29,'VBN':30,'VBP':31,'VBZ':32,'WDT':33,\
    'WP':34,'WP$':35,'WRB':36, 'END':37}
TAG_IDS = {v:k for k,v in TAGS.items()}
NUM_TAGS = len(TAGS)
#PRUNED_PUNCTUATION = '!""#\\\'\'()*+,--./:;<=>?[]^_``{|}~'
PUNCTUATION_TAGS = {'#':'#','\'\'':'\'\'','(':'(','{':'(',')':')','}':')',',':',','!':'.','.':'.','?':'.','-':':','--':':','...':':',':':':',';':':','`':'``','``':'``','non-``':'``'}


## ======================== Loading Data ======================== ##

def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.
    
    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    sentences = []
    sentence = ['O']
    sentences_tags = None

    with open(sentence_file, 'r') as x_data:
        print(sentence_file)
        next(x_data) # Skip first -DOCSTART-
        next(x_data)
        for line in x_data:
            word = line.split(',',1)[1].strip()[1:-1]
            if word.strip() == '-DOCSTART-':
                sentence.append("END")
                sentences.append(sentence)
                sentence = ['O']
            else: 
                sentence.append(word)
        sentence.append('END')
        sentences.append(sentence)
    if tag_file:
        print(tag_file)
        sentences_tags = []
        sentence_tags = ['O']
        with open(tag_file, 'r') as y_data:
            next(y_data)
            next(y_data)
            for line in y_data:
                tag = line.split(',',1)[1].strip()[1:-1]
                if tag == 'O':
                    sentence_tags.append('END')
                    sentences_tags.append(sentence_tags)
                    sentence_tags = ['O']
                else: 
                    sentence_tags.append(tag)
            sentence_tags.append('END')
            sentences_tags.append(sentence_tags)

    return sentences, sentences_tags


def prune_data(sentences, tags):
    pruned_sentences = []
    pruned_tags = []

    for sentence, tag in zip(sentences,tags):
    #df = pd.DataFrame({'word':sentence, 'tag':tag}, columns=['word','tag'])
        new_sentence = [word for word,label in zip(sentence,tag) if label in TAGS.keys()]
        new_tag = [label for word,label in zip(sentence,tag) if label in TAGS.keys()]
        pruned_sentences.append(new_sentence)
        pruned_tags.append(new_tag)
    return pruned_sentences, pruned_tags

def prune_sentences(sentences):
    pruned_sentences = [[word for word in sentence if (word not in PUNCTUATION_TAGS.keys()) and not "$" in word] for sentence in sentences]
    return pruned_sentences
        


## ======================== Smoothing ======================== ##

def rare_words_train(sentences, threshold = 5):
    word_counts = {}
    for sentence in sentences:
        for word in sentence:
            word_counts[word] = word_counts.get(word,0) + 1
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = []
        for word in sentence:
            if word_counts[word] < threshold:
                processed_sentence.append("_RARE_")
            else: processed_sentence.append(word)
        processed_sentences.append(processed_sentence)
    return processed_sentences, word_counts

def rare_words_morpho_train(sentences, threshold = 5):
    word_counts = {}
    for sentence in sentences:
        for word in sentence:
            word_counts[word] = word_counts.get(word,0) + 1
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = []
        for word in sentence:
            if word_counts[word] < threshold: # from https://stathwang.github.io/part-of-speech-tagging-with-trigram-hidden-markov-models-and-the-viterbi-algorithm.html     
                if not re.search(r'\w', word): 
                    processed_sentence.append('_PUNCS_')
                elif re.search(r'[A-Z]', word):
                    processed_sentence.append('_CAPITAL_')
                elif re.search(r'\d', word):
                    processed_sentence.append('_NUM_')
                elif re.search(r'(ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)',word):
                    processed_sentence.append('_NOUNLIKE_')
                elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem)', word):
                    processed_sentence.append('_VERBLIKE_')
                elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)',word):
                    processed_sentence.append('_ADJLIKE_')
                else:
                    processed_sentence.append('_RARE_')
            else: processed_sentence.append(word)
        processed_sentences.append(processed_sentence)
    return processed_sentences, word_counts

def rare_words(sentences, word_counts, threshold = 5):
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = []
        for word in sentence:
            if word_counts.get(word,0) < threshold:
                processed_sentence.append("_RARE_")
            else: processed_sentence.append(word)
        processed_sentences.append(processed_sentence)
    return processed_sentences

def rare_words_morpho(sentences, word_counts, threshold = 5):
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = []
        for word in sentence:
            if word_counts.get(word,0) < threshold: # from https://stathwang.github.io/part-of-speech-tagging-with-trigram-hidden-markov-models-and-the-viterbi-algorithm.html
                if not re.search(r'\w', word): 
                    processed_sentence.append('_PUNCS_')
                elif re.search(r'[A-Z]', word):
                    processed_sentence.append('_CAPITAL_')
                elif re.search(r'\d', word):
                    processed_sentence.append('_NUM_')
                elif re.search(r'(ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)',word):
                    processed_sentence.append('_NOUNLIKE_')
                elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem)', word):
                    processed_sentence.append('_VERBLIKE_')
                elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)',word):
                    processed_sentence.append('_ADJLIKE_')
                else:
                    processed_sentence.append('_RARE_')
            else: processed_sentence.append(word)
        processed_sentences.append(processed_sentence)
    return processed_sentences



def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.
    
    """
    pass



class POSTagger():


    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        #trigram_transmissions = np.zeros(NUM_TAGS,NUM_TAGS,NUM_TAGS) # q(C|A,B) = t_t[id[A],id[B],id[C]]
        ngram_transmissions = None
        emissions = None
        word_encodings = None


    def encode_words(self, sentences):
        """
        @returns: { words : int } encoding of words 
        """       
        index = 0
        word_encodings = {}
        for sentence in sentences:
            for word in sentence:
                if not word in word_encodings.keys():
                    word_encodings[word] = index
                    index += 1
        return word_encodings


    def ngram_counter(self, sentences, tags, n):
        """
        @params: 
            - sentences, tags: 2d array of data
            - n: length of pos sequence
        @return: 
            - dict: {[pos_1,..,pos_n] : count} - number of times a pos sequence appears 
        """   
        counter = {}

        for sentence, tag in zip(sentences, tags):
            assert(len(sentence) == len(tag))   #housekeeping check
            l = len(sentence)
            ngram_range = range(1, l-1) if n > 1 else range(l)       #skip the only START and END case except for unigrams

            for i in ngram_range:      
                ngram = None
                if i >= n-1 and i <= l-n: 
                    ngram_labels = tuple(tag[i-n+1:i+1])
                elif i < n-1: 
                    ngram_labels = tuple(['O' for x in range(n-i-1)] + tag[:i+1])
                    #print(str(tag[:i+1]) + " " + str(tag[i+1:n]))
                    #print(ngram_labels)
                else:
                    ngram_labels = tuple(tag[i:]+['END' for x in range(i+n-l)])
                    #print(str(i)+ " "+str(sentence_len))
                    #print(ngram_labels)
                
                assert(len(ngram_labels) == n)
                if(n==1): labels = TAGS[ngram_labels[0]]
                else: labels = tuple([TAGS[t] for t in ngram_labels])
                counter[labels] = counter.get(labels, 0) + 1

        #pprint.pprint(counter)
        return counter


    def tag_assignment_counter(self, sentences, tags): 
        """
        @params: 
            - sentences, tags: 2d array of data 
        @return: 
            - dict: {[word, pos] : count} - number of times a word is assigned with some tag
        """        
        counter = {}
        for sentence, tag in zip(sentences, tags):
            assert(len(sentence) == len(tag))
            for word, label in zip(sentence, tag):
                counter[(word, TAGS[label])] = counter.get((word, TAGS[label]), 0) + 1
        #pprint.pprint(counter)
        return counter


    def get_transmissions(self, data, n = 3):
        """
        @return: n-d array representing transmission matrix 
                q(s|u,v) = c(u,v,s) / c(u,v)  => arr[u,v,s]
        """        
        matrix_shape = tuple(NUM_TAGS for i in range(n))
        trans_matrix = np.zeros(matrix_shape)
        ngrams = [self.ngram_counter(data[0],data[1],i) for i in range(1,n+1)]
        if n==2:
            for u in range(NUM_TAGS):
                c_u = ngrams[0].get((u),0)
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)
                    trans_matrix[u][v] = c_uv / c_u if c_u != 0 else 0
        elif n==3:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)     
                    for s in range(NUM_TAGS):
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        trans_matrix[u][v][s] = c_uvs / c_uv if c_uv != 0 else 0
        elif n==4:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    for s in range(NUM_TAGS):
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        for w in range(NUM_TAGS):
                            c_uvsw = ngrams[3].get((u,v,s,w),0)
                            trans_matrix[u][v][s][w] = c_uvsw / c_uvs if c_uvs != 0 else 0
        #print(trans_matrix)
        return trans_matrix

    def get_transmissions_good_turing(self, data, n=3): 
        matrix_shape = tuple(NUM_TAGS for i in range(n))
        trans_matrix = np.zeros(matrix_shape)
        ngrams = []
        for i in range(1,n+1):
            igram_counter = self.ngram_counter(data[0], data[1], i)
            igram_freq = Counter(igram_counter.values())
            igram_smoothed = { x : (igram_counter[x]+1) * igram_freq[(igram_counter[x]+1)] / igram_freq[igram_counter[x]] for x in igram_counter.keys() }
            ngrams.append(igram_smoothed)
        if n==2:
            for u in range(NUM_TAGS):
                c_u = ngrams[0].get((u),0)
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)
                    trans_matrix[u][v] = c_uv / c_u if c_u != 0 else 0
        elif n==3:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)     
                    for s in range(NUM_TAGS):
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        trans_matrix[u][v][s] = c_uvs / c_uv if c_uv != 0 else 0
        elif n==4:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    for s in range(NUM_TAGS):
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        for w in range(NUM_TAGS):
                            c_uvsw = ngrams[3].get((u,v,s,w),0)
                            trans_matrix[u][v][s][w] = c_uvsw / c_uvs if c_uvs != 0 else 0
        return trans_matrix     

    def get_transmissions_add_k(self, data, k, n=3):
        matrix_shape = tuple(NUM_TAGS for i in range(n))
        trans_matrix = np.zeros(matrix_shape)
        ngrams = [self.ngram_counter(data[0],data[1],i) for i in range(1,n+1)]
        if n==2:
            for u in range(NUM_TAGS):
                c_u = ngrams[0].get((u),0)
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)
                    trans_matrix[u][v] = (c_uv+k) / (c_u+k**(n-1))
        elif n==3:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)     
                    for s in range(NUM_TAGS):
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        trans_matrix[u][v][s] = (c_uvs+k) / (c_uv+k*NUM_TAGS**(n-1))
        elif n==4:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    for s in range(NUM_TAGS):
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        for w in range(NUM_TAGS):
                            c_uvsw = ngrams[3].get((u,v,s,w),0)
                            trans_matrix[u][v][s][w] = (c_uvsw+k) / (c_uvs+k*NUM_TAGS**(n-1))
        return trans_matrix


    def get_transmissions_linear_interpolation(self, data, l_values, n=3):
        """
        @params: 
            -data: sentences, tags
            -l_values [l1, l2,...] of length n-1
            -l1,l2>=0, l1+l2<=1

        """
        matrix_shape = tuple(NUM_TAGS for i in range(n))
        trans_matrix = np.zeros(matrix_shape)
        ngrams = [self.ngram_counter(data[0],data[1],i) for i in range(1,n+1)]
        if n==2:
            for u in range(NUM_TAGS):
                c_u = ngrams[0].get((u),0)
                for v in range(NUM_TAGS):
                    c_v = ngrams[0].get((v),0)
                    c_uv = ngrams[1].get((u,v), 0)
                    bigram_transmission = c_uv / c_u if c_u != 0 else 0
                    unigram_transmission = c_v / len(data[1])
                    trans_matrix[u][v] = l_values[0]* bigram_transmission + (1-sum(l_values)) * unigram_transmission
        elif n==3:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)
                    c_v = ngrams[0].get((v),0)
                    for s in range(NUM_TAGS):
                        c_s = ngrams[0].get((s),0)
                        c_vs = ngrams[1].get((v,s),0)
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        trigram_transmission = c_uvs / c_uv if c_uv != 0 else 0
                        bigram_transmission = c_vs / c_v if c_v != 0 else 0
                        unigram_transmission = c_s / len(data[1]) #I think this is right but not 100% sure
                        trans_matrix[u][v][s] = l_values[0]* trigram_transmission + l_values[1] * bigram_transmission + (1-sum(l_values)) * unigram_transmission
        elif n==4:
            for u in range(NUM_TAGS):
                for v in range(NUM_TAGS):
                    c_uv = ngrams[1].get((u,v), 0)
                    c_v = ngrams[0].get((v),0)
                    for s in range(NUM_TAGS):
                        c_s = ngrams[0].get((s),0)
                        c_vs = ngrams[1].get((v,s),0)
                        c_uvs = ngrams[2].get((u,v,s), 0) 
                        for w in range(NUM_TAGS):
                            c_uvsw = ngrams[3].get((u,v,s,w),0)
                            c_vsw = ngrams[2].get((v,s,w),0)
                            c_sw = ngrams[1].get((s,w),0)
                            c_w = ngrams[0].get((w),0)
                            fourgram_transmission = c_uvsw / c_uvs if c_uvs != 0 else 0
                            trigram_transmission = c_vsw / c_vs if c_vs != 0 else 0
                            bigram_transmission = c_sw / c_s if c_s != 0 else 0
                            unigram_transmission = c_w / len(data[1]) #I think this is right but not 100% sure
                            trans_matrix[u][v][s][w] = l_values[0]* fourgram_transmission + l_values[1] * trigram_transmission + l_values[2] * bigram_transmission + (1-sum(l_values)) * unigram_transmission

        #print(trans_matrix)
        return trans_matrix


    def get_emissions(self, data):
        """
        @return: 2d array of shape (number of unique words, number of pos = 36) representing emission matrix
                e(x|s) = c(s->x) / c(s) => arr[x,s]
        """        
        self.word_encodings = self.encode_words(data[0])
        unigram = self.ngram_counter(data[0], data[1], 1)
        tag_assignment = self.tag_assignment_counter(data[0], data[1])
        num_words = len(self.word_encodings)
        emission_matrix = np.zeros((num_words, NUM_TAGS))

        for sentence, tag in zip(data[0], data[1]):
            for word, label in zip(sentence, tag):
                x = self.word_encodings[word]
                s = TAGS[label]
                emission_matrix[x][s] = tag_assignment.get((word,s), 0) / unigram[(s)]
        #print(emission_matrix)
        return emission_matrix


    def train(self, data, smoothing = 'None', n = 3, k=.3, l_values=[.5,.3]):
        """Trains the model by computing transition and emission probabilities.
        Set transmission and emission matrix with customized smoothing techniques       
        """
        if smoothing == 'add-k':
            self.ngram_transmissions = self.get_transmissions_add_k(data,k,n)
        elif smoothing == 'linear_interpolation':
            self.ngram_transmissions = self.get_transmissions_linear_interpolation(data, l_values, n)
        else: 
            self.ngram_transmissions = self.get_transmissions(data,n)
        
        self.emissions = self.get_emissions(data)


    def sequence_probability(self, sequence, tags, n=3):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        probability = np.log(1)
        sequence = ['O']*(n-2) + sequence 
        tags = ['O']*(n-2) + tags 
        if n==2:
            for i in range(1, len(sequence)):
                probability += np.log(self.ngram_transmissions[TAGS[tags[i-1]]][TAGS[tags[i]]])
                probability += np.log(self.emissions[self.word_encodings[sequence[i]]][TAGS[tags[i]]])

        elif n==3:
            for i in range(2, len(sequence)):
                probability += np.log(self.ngram_transmissions[TAGS[tags[i-2]]][TAGS[tags[i-1]]][TAGS[tags[i]]])
                probability += np.log(self.emissions[self.word_encodings[sequence[i]]][TAGS[tags[i]]])
        elif n==4:
            for i in range(3, len(sequence)):
                probability += np.log(self.ngram_transmissions[TAGS[tags[i-3]]][TAGS[tags[i-2]]][TAGS[tags[i-1]]][TAGS[tags[i]]])
                probability += np.log(self.emissions[self.word_encodings[sequence[i]]][TAGS[tags[i]]])

        return probability


    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        return []

    def beam_search(self, sequence, K=1, n=3): 
        prev_indices = [0]
        prev_pis = [0]
        assigned_tags = [0]

        for k in range(1,len(sequence)):
            new_indices = []
            new_pis = []
            for index, pi in zip(prev_indices, prev_pis):
                u = index // NUM_TAGS
                v = index % NUM_TAGS
                for i in range(NUM_TAGS):
                    new_pi = pi + np.log(self.ngram_transmissions[u][v][i]) + np.log(self.emissions[self.word_encodings[sequence[k]]][i])
                    new_indices.append(v*NUM_TAGS + i)
                    new_pis.append(new_pi)
            
            new_sorted = sorted(zip(new_pis, new_indices), key=lambda pair:pair[0])[-K:]
            prev_indices = [i for p, i in new_sorted]
            prev_pis = [p for p, i in new_sorted]
            assigned_tags.append(prev_indices[-1] % NUM_TAGS)

        return [TAG_IDS[tag] for tag in assigned_tags]


    def viterbi(self, sequence):
        """Generates tags through Viterbi algorithm
        
        """
        # print(sequence)
        lattice = np.zeros((NUM_TAGS**2,len(sequence)))
        backpointers = np.zeros((NUM_TAGS**2, len(sequence)))
        lattice[0][0] = 1 #u,v,k -> NUM_TAGS*u+v,k
        lattice = np.log(lattice)
        nonzero_indices = [0] #keep track of the indices with nonzero pi values from the previous time step
        for k in range(1,len(sequence)):
            maxes = {} #keep track of nonzero pi for current step
            for index in nonzero_indices:
                u = index // NUM_TAGS
                v = index % NUM_TAGS
                prev_pi = lattice[index,k-1]
                # print(prev_pi)
                for i in range(NUM_TAGS):
                    # print(f"pi = {prev_pi} + {np.log(self.trigram_transmissions[u][v][i])} + {np.log(self.emissions[self.word_encodings[sequence[k]]][i])}")
                    pi = prev_pi + np.log(self.ngram_transmissions[u][v][i]) + np.log(self.emissions[self.word_encodings[sequence[k]]][i])
                    if pi > np.NINF:
                        if(v, i) not in maxes.keys():
                            maxes[(v,i)] = [(u, pi)]
                        else:
                            maxes[(v,i)].append((u,pi))

            nonzero_indices = []
            # print(maxes)
            for (v,w) in maxes.keys(): #find best path for each node and update the lattice and backpointers
                best_path = max(maxes[(v,w)], key=lambda x: x[1])
                # print(best_path)
                u = best_path[0] 
                lattice[v*NUM_TAGS+w][k] = best_path[1]
                backpointers[v*NUM_TAGS+w][k] = u*NUM_TAGS+v
                if best_path[1]>np.NINF:
                    nonzero_indices.append(v*NUM_TAGS+w)
            # print(nonzero_indices)
        endpoints = []
        for i in range(NUM_TAGS-1,NUM_TAGS**2-1,NUM_TAGS):
            endpoints.append((i,lattice[i,len(sequence)-1]))
        # print(endpoints)
        best_endpoint = max(endpoints, key= lambda x: x[1])
        # print(best_endpoint)
        # print(lattice)
        np.savetxt('lattice.csv',lattice,delimiter=',')
        # print(backpointers)
        tags = [best_endpoint[0]//NUM_TAGS,NUM_TAGS-1] # Initializes tags with end tag and best preceding tag
        for k in range(len(sequence)-1,1,-1):
            # print(tags)
            prev_index = int(backpointers[tags[0]*NUM_TAGS+tags[1]][k])
            tags.insert(0,prev_index//NUM_TAGS)
        tag_strings = [TAG_IDS[tag] for tag in tags]
        return tag_strings

def deprune_output(original_sentences, tags):
    for sentence, tag in zip(original_sentences, tags):
        for i in range(len(sentence)):
            if sentence[i] in string.punctuation:
                tag.insert(i,sentence[i])

def deprune_formatted_output(original_tags, tags):
    for i in range(len(original_tags)):
        if original_tags[i] not in TAGS.keys():
            tags.insert(i,original_tags[i])

def deprune_formatted_output_from_sentences(original_sentences, tags):
    for i in range(len(original_sentences)):
        if original_sentences[i] in PUNCTUATION_TAGS.keys():
            tags.insert(i,PUNCTUATION_TAGS[original_sentences[i]])
        elif "$" in original_sentences[i]: 
            tags.insert(i,"$")

def format_output(tags):
    tag_list = [tag for line in tags for tag in line if tag != 'END']
    for i in range(len(tag_list)):
        if tag_list[i] == 'START':
            tag_list[i] = 'O'
    return tag_list

def format_output_sentences(sentences):
    words = [word for line in sentences for word in line if word != 'END']
    for i in range(len(words)):
        if words[i] == 'START':
            words[i] = '-DOCSTART-'
    return words






if __name__ == "__main__":
    # TODO: compare pruning techniques
    # figure out punctuation cleaning without tags?
    # 

    train_x,train_y = load_data("data/train_x.csv", "data/train_y.csv")
    train_x_pruned, train_y_pruned = prune_data(train_x, train_y)
    train_x_rare, word_counts = rare_words_morpho_train(train_x_pruned)
    for i in range(len(train_x_rare)):
        if len(train_x_rare[i]) != len(train_y_pruned[i]): print(len(train_x_rare[i]), len(train_y_pruned[i]))
    dev_x, dev_y = load_data("data/dev_x.csv", "data/dev_y.csv")
    dev_x_pruned, dev_y_pruned = prune_data(dev_x,dev_y)
    dev_x_rare = rare_words_morpho(dev_x_pruned,word_counts, 5)
    mini_x, mini_y = load_data("data/mini_x.csv", "data/mini_y.csv")
    mini_x_pruned = prune_sentences(mini_x)
    mini_x_rare = rare_words_morpho(mini_x_pruned,word_counts, 5)
    pos_tagger = POSTagger()
    pos_tagger.train([train_x_rare, train_y_pruned], smoothing='add-k',k=.05)

    mini_y_pred = [pos_tagger.viterbi(sentence) for sentence in mini_x_rare]
    pd.DataFrame(mini_x_rare).to_csv('data/mini_x_rare.csv')
    pd.DataFrame(mini_y_pred).to_csv('data/mini_y_pred.csv')
    mini_y_pred_tags = format_output(mini_y_pred)
    print(len(mini_y_pred_tags))
    deprune_formatted_output_from_sentences(format_output_sentences(mini_x),mini_y_pred_tags)
    print('depruned')
    print(len(mini_y_pred_tags))
    pd.DataFrame(enumerate(mini_y_pred_tags),columns=['id','tag']).to_csv('data/mini_y_pred_tags.csv',index=False, quoting=csv.QUOTE_NONNUMERIC)
    # for i in range(len(mini_y_pred)):
    #     print(len(mini_y_pred[i]),len(mini_y[i]))
    # dev_y_pred = [pos_tagger.viterbi(sentence) for sentence in dev_x_rare]
    # =================================================================TEST SUBOPTIMALITIES================================================================
    # probabilities = []
    # for i in range(len(dev_x)):
    #     y_pred_prob = pos_tagger.sequence_probability(dev_x_rare[i],dev_y_pred[i])
    #     y_prob = pos_tagger.sequence_probability(dev_x_rare[i],dev_y_pruned[i])
    #     if y_prob > y_pred_prob:
    #         print("oh no")
    #         print(y_pred_prob,y_prob,i,len(dev_x_rare[i]))

    #     probabilities.append((y_pred_prob,y_prob))
    #=======================================================================================================================================================
    
    
    # dev_y_pred_tags = format_output(dev_y_pred)
    # deprune_output(format_output(dev_y),dev_y_pred_tags)
    # for i in range(len(dev_y_pred)):
    #     if len(dev_y_pred_tags[i]) != len(dev_y[i]): print(len(dev_y_pred_tags[i]),len(dev_y[i]),i)
    # dev_y_pred_df = pd.DataFrame(enumerate(dev_y_pred_tags),columns=['id','tag'])
    # dev_y_pred_df.to_csv('data/dev_y_pred.csv',index=False, quoting=csv.QUOTE_NONNUMERIC)
    # dev_y_test_df = pd.DataFrame(enumerate(format_output(dev_y)),columns=['id','tag'])
    # dev_y_test_df.to_csv('data/dev_y_test.csv',index=False, quoting=csv.QUOTE_NONNUMERIC)
    print('done')
    # print(test_beam)
    #pos_tagger.ngram_counter(train_data[0],train_data[1], 3)
    #pos_tagger.tag_assignment_counter(train_data[0], train_data[1])


    # Experiment with your decoder using greedy decoding, beam search, viterbi...


    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc
    #evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    # test_predictions = []
    # for sentence in test_data:
    #     test_predictions.extend(pos_tagger.inference(sentence)
    
    # Write them to a file to update the leaderboard
    # TODO
