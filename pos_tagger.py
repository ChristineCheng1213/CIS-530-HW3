""" Contains the part of speech tagger class. """
import numpy as np
import string
import copy
import sys

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
        for line in x_data:
            word = line.split(',')[1].strip()[1:-1]
            if word.strip() == '-DOCSTART-':
                sentence.append("END")
                sentences.append(sentence)
                sentence = ['START']
            elif word in string.punctuation:
                continue
            else: 
                sentence.append(word)
    
    if tag_file:
        print(tag_file)
        sentences_tags = []
        sentence_tags = ['^']
        with open(tag_file, 'r') as y_data:
            next(y_data)
            for line in y_data:
                tag = line.split(',')[1].strip()[1:-1]
                #print(tag)                
                if tag == 'O':
                    sentence_tags.append('$')
                    sentences_tags.append(sentence_tags)
                    sentence_tags = ['^']
                elif tag in string.punctuation: 
                    continue
                else: 
                    sentence_tags.append(tag)

    # print(sentences)
    # print(sentence_tags)
    return sentences, sentences_tags

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
    tag_ids = {'^':0,'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6,'JJ':7,'JJR':8,'JJS':9,'LS':10,'MD':11,'NN':12,\
        'NNS':13,'NNP':14,'NNPS':15,'PDT':16,'POS':17,'PRP':18,'PRP$':19,'RB':20,'RBR':21,'RBS':22,'RP':23,\
        'SYM':24,'TO':25,'UH':26,'VB':27,'VBD':28,'VBG':29,'VBN':30,'VBP':31,'VBZ':32,'WDT':33,\
        'WP':34,'WP$':35,'WRB':36, '$':37}
    NUM_TAGS = len(tag_ids)

    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        #trigram_transmissions = np.zeros(NUM_TAGS,NUM_TAGS,NUM_TAGS) # q(C|A,B) = t_t[id[A],id[B],id[C]]
        trigram_transmissions = None
        emissions = None

    """
    @params: 
        - sentences, tags: 2d array of data
        - n: length of pos sequence
    @return: 
        - dict: {[pos_1,..,pos_n] : count} - number of times a pos sequence appears 
    """
    def ngram_counter(self, sentences, tags, n):
        counter = {}
        bigger_s = []
        bigger_t = []
        for j, (sentence, tag) in enumerate(zip(sentences, tags)):

            if len(sentence) < len(tag): bigger_t.append(j)
            if len(sentence) > len(tag): bigger_s.append(j)

            # sentence_len = len(sentence)
            
            # for i in range(1,sentence_len):
            #     ngram = None
            #     if i >= n-1 and i <= sentence_len-n: 
            #         ngram_labels = tuple(tag[i-n+1:i+1])
            #     elif i < n-1: 
            #         ngram_labels = tuple(['^' for x in range(n-i-1)] + tag[:i+1])
            #         #print(str(tag[:i+1]) + " " + str(tag[i+1:n]))
            #     else:
            #         #print(str(i)+ " "+str(sentence_len))
            #         ngram_labels = tuple(tag[i:])
            #         #print(tag[i:])
                    
            #     counter[ngram_labels] = counter.get(ngram_labels, 0)+1
        print(bigger_s)
        print(bigger_t)
        return counter

    """
    @params: 
        - sentences, tags: 2d array of data 
    @return: 
        - dict: {[word, pos] : count} - number of times a word is assigned with some tag
    """
    def tag_assignment_counter(self, sentences, tags): 
        pass

    """
    @return: n-d array representing transmission matrix 
    """
    def get_transmissions(self, data):
        pass

    """
    @return: 2d array of shape (number of unique words, number of pos = 36) representing emission matrix
    """
    def get_emissions(self, data):
        pass

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
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    # dev_data, dev_tags = load_data("data/dev_x.csv", "data/dev_y.csv")
    # test_data, test_tags = load_data("data/test_x.csv")

    #pos_tagger.train(train_data)

    pos_tagger.ngram_counter(train_data[0],train_data[1], 4)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    #evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    # test_predictions = []
    # for sentence in test_data:
    #     test_predictions.extend(pos_tagger.inference(sentence)
    
    # Write them to a file to update the leaderboard
    # TODO
