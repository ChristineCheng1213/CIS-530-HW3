""" Contains the part of speech tagger class. """
import numpy as np
import pandas as pd
import string
import copy
import sys
import pprint

TAGS = {'START':0,'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6,'JJ':7,'JJR':8,'JJS':9,'LS':10,'MD':11,'NN':12,\
    'NNS':13,'NNP':14,'NNPS':15,'PDT':16,'POS':17,'PRP':18,'PRP$':19,'RB':20,'RBR':21,'RBS':22,'RP':23,\
    'SYM':24,'TO':25,'UH':26,'VB':27,'VBD':28,'VBG':29,'VBN':30,'VBP':31,'VBZ':32,'WDT':33,\
    'WP':34,'WP$':35,'WRB':36, 'END':37, 'UNKOWN':38}
NUM_TAGS = len(TAGS)


def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.
    
    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    sentences = []
    sentence = ['START']
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
                sentence = ['START']
            else: 
                sentence.append(word)
    
    if tag_file:
        print(tag_file)
        sentences_tags = []
        sentence_tags = ['START']
        with open(tag_file, 'r') as y_data:
            next(y_data)
            next(y_data)
            for line in y_data:
                tag = line.split(',',1)[1].strip()[1:-1]
                if tag == 'O':
                    sentence_tags.append('END')
                    sentences_tags.append(sentence_tags)
                    sentence_tags = ['START']
                else: 
                    sentence_tags.append(tag)

    if tag_file: return prune_data(sentences, sentences_tags)
    else: return sentences, sentences_tags


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
        word_encodings = None
        trigram_transmissions = None
        emissions = None

    def generate_states(self, data):
        self.word_encodings = self.encode_words(data[0])
        self.trigram_transmissions = self.get_transmissions(data)
        self.emissions = self.get_emissions(data)


    """
    @returns: { words : int } encoding of words 
    """
    def encode_words(self, sentences):
        index = 0
        word_encodings = {}
        for sentence in sentences:
            for word in sentence:
                if not word in word_encodings.keys():
                    word_encodings[word] = index
                    index += 1
        return word_encodings

    """
    @params: 
        - sentences, tags: 2d array of data
        - n: length of pos sequence
    @return: 
        - dict: {[pos_1,..,pos_n] : count} - number of times a pos sequence appears 
    """
    def ngram_counter(self, sentences, tags, n):
        counter = {}

        for sentence, tag in zip(sentences, tags):
            assert(len(sentence) == len(tag))   #housekeeping check
            l = len(sentence)
            ngram_range = range(1, l-1) if n > 1 else range(l)      #skip the only START and END case except for unigrams

            for i in ngram_range:      
                ngram = None
                if i >= n-1 and i <= l-n: 
                    ngram_labels = tuple(tag[i-n+1:i+1])
                elif i < n-1: 
                    ngram_labels = tuple(['START' for x in range(n-i-1)] + tag[:i+1])
                    #print(str(tag[:i+1]) + " " + str(tag[i+1:n]))
                    #print(ngram_labels)
                else:
                    ngram_labels = tuple(tag[i:]+['END' for x in range(i+n-l)])
                    #print(str(i)+ " "+str(sentence_len))
                    #print(ngram_labels)
                
                assert(len(ngram_labels) == n)
                labels = tuple([TAGS[t] for t in ngram_labels])
                counter[labels] = counter.get(labels, 0) + 1

        #pprint.pprint(counter)
        return counter

    """
    @params: 
        - sentences, tags: 2d array of data 
    @return: 
        - dict: {[word, pos] : count} - number of times a word is assigned with some tag
    """
    def tag_assignment_counter(self, sentences, tags): 
        counter = {}
        for sentence, tag in zip(sentences, tags):
            assert(len(sentence) == len(tag))
            for word, label in zip(sentence, tag):
                counter[(word, TAGS[label])] = counter.get((word, TAGS[label]), 0) + 1
        #pprint.pprint(counter)
        return counter

    """
    @return: n-d array representing transmission matrix 
             q(s|u,v) = c(u,v,s) / c(u,v)  => arr[u,v,s]
    """
    def get_transmissions(self, data):
        trigrams = self.ngram_counter(data[0], data[1], 3)
        bigrams = self.ngram_counter(data[0], data[1], 2)
        trans_matrix = np.zeros((NUM_TAGS,NUM_TAGS,NUM_TAGS))

        for u in range(NUM_TAGS):
            for v in range(NUM_TAGS):
                c_uv = bigrams.get((u,v), 0)     # TODO: How to deal with 0 values 
                for s in range(NUM_TAGS):
                    c_uvs = trigrams.get((u,v,s), 0) 
                    trans_matrix[u][v][s] = c_uvs / c_uv if c_uv != 0 else 0

        #print(trans_matrix)
        return trans_matrix
    def get_transmissions_add_k(self, data, k):
        trigrams = self.ngram_counter(data[0], data[1], 3)
        bigrams = self.ngram_counter(data[0], data[1], 2)
        trans_matrix = np.zeros((NUM_TAGS,NUM_TAGS,NUM_TAGS))

        for u in range(NUM_TAGS):
            for v in range(NUM_TAGS):
                c_uv = bigrams.get((u,v), 0)     
                for s in range(NUM_TAGS):
                    c_uvs = trigrams.get((u,v,s), 0) 
                    trans_matrix[u][v][s] = (c_uvs+k) / (c_uv+k*NUM_TAGS*NUM_TAGS) #add k to numerator, k*number of possible bigrams to denominator

        #print(trans_matrix)
        return trans_matrix
    def get_transmissions_linear_interpolation(self, data, l1, l2):
        """
        @params: 
            -data: sentences, tags
            -l1: weight for trigram
            -l2: weight for bigram
            -l1,l2>=0, l1+l2<=1

        """
        trigrams = self.ngram_counter(data[0], data[1], 3)
        bigrams = self.ngram_counter(data[0], data[1], 2)
        unigrams = self.ngram_counter(data[0],data[1],1)
        trans_matrix = np.zeros((NUM_TAGS,NUM_TAGS,NUM_TAGS))

        for u in range(NUM_TAGS):
            for v in range(NUM_TAGS):
                c_uv = bigrams.get((u,v), 0)
                c_vs = bigrams.get((v,s),0)
                c_v = unigrams.get((v),0)
                for s in range(NUM_TAGS):
                    c_uvs = trigrams.get((u,v,s), 0) 
                    trigram_transmission = c_uvs / c_uv if c_uv != 0 else 0
                    bigram_transmission = c_vs / c_v if c_v != 0 else 0
                    unigram_transmission = c_v / len(data[1]) #I think this is right but not 100% sure
                    trans_matrix[u][v][s] = l1* trigram_transmission + l2 * bigram_transmission + (1-l1-l2) * unigram_transmission

        #print(trans_matrix)
        return trans_matrix
    """
    @return: 2d array of shape (number of unique words, number of pos = 36) representing emission matrix
             e(x|s) = c(s->x) / c(s) => arr[x,s]
    """
    def get_emissions(self, data):
        unigram = self.ngram_counter(data[0], data[1], 1)
        tag_assignment = self.tag_assignment_counter(data[0], data[1])
        num_words = len(self.word_encodings)
        emission_matrix = np.zeros((num_words, NUM_TAGS))
        #print(unigram)

        for sentence, tag in zip(data[0], data[1]):
            for word, label in zip(sentence, tag):
                x = self.word_encodings[word]
                s = TAGS[label]
                emission_matrix[x][s] = tag_assignment.get((word,s), 0) / unigram[tuple([s])]
        #print(emission_matrix)
        return emission_matrix


    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        pass

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        return 0.

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        return []


if __name__ == "__main__":


    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    # dev_data, dev_tags = load_data("data/dev_x.csv", "data/dev_y.csv")
    # test_data, test_tags = load_data("data/test_x.csv")
    pos_tagger = POSTagger()
    pos_tagger.generate_states(train_data)
    #pos_tagger.train(train_data)
    #pos_tagger.get_transmissions(train_data)
    #pos_tagger.get_emissions(train_data)

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
