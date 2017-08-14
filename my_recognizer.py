import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    sequences = test_set.get_all_sequences()
    sequences_len = test_set.get_all_Xlengths()

    for sequence in sequences:
        prob = {}
        guess = None

        x, length = sequences_len[sequence]
        # find the best word match and update guess
        for word, model in models.items():
            try:
                prob[word] = model.score(x, length)
                guess = word if (guess is None or prob[word] > prob[guess]) else guess
            except:
                prob[word] = float('-inf')

        probabilities += [prob]
        guesses += [guess]

    return probabilities, guesses
