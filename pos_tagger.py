""" Contains the part of speech tagger class. """
import numpy as np
NUM_TAGS = 36
def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.
    
    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    # TODO: Prune punctuation, maybe use dataframes?
    sentences = []
    sentence = ['START']
    sentence_tags = None
    vocabulary = set()
    with open(sentence_file, 'r') as x_data:
        next(x_data) # Skip first -DOCSTART-
        for line in x_data:
            word = line.split(',')[1]
            if word == "-DOCSTART-":
                sentence.append("END")
                sentences.append(sentence)
                sentence = ['START']
            else: 
                sentence.append(word)
                vocabulary.add(word)
    if(tag_file):
        sentences_tags = []
        sentence_tags = ['^']
        with open(tag_file, 'r') as y_data:
            next(y_data)
            for line in y_data:
                tag = line.split(',')[1]
                if word == "O":
                    sentence_tags.append('$')
                    sentences_tags.append(sentence_tags)
                    sentence_tags = ['^']
                else: sentence_tags.append(tag)


    return sentences, sentences_tags, vocabulary

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
        tag_ids = {'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6,'JJ':7,'JJR':8,'JJS':9,'LS':10,'MD':11,'NN':12,\
            'NNS':13,'NNP':14,'NNPS':15,'PDT':16,'POS':17,'PRP':18,'PRP$':19,'RB':20,'RBR':21,'RBS':22,'RP':23,\
            'SYM':24,'TO':25,'UH':26,'VB':27,'VBD':28,'VBG':29,'VBN':30,'VBP':31,'VBZ':32,'WDT':33,\
            'WP':34,'WP$':35,'WRB':36}
        trigram_transmissions = np.zeros(NUM_TAGS,NUM_TAGS,NUM_TAGS) # q(C|A,B) = t_t[id[A],id[B],id[C]]
        emissions = np.zeros(vocab_size,NUM_TAGS)

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
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence)
    
    # Write them to a file to update the leaderboard
    # TODO
