import math
import os
import random
import random as rand
from collections import Counter

import numpy as np

import language


# ===================================== TOY ============================================================================
def generate_toy_data(language_name="toy_synthetic", num_train=1e6):
    # Create a language
    mylang = language.Language()

    # Set the phonology of the language
    phonemes = {
        "C": ["p", "b", "t", "d", "k", "g", "m", "n", "f", "v", "s", "z", "h", "j", "w", "r", "l", "c", "x"],
        "V": ["a", "e", "i", "o", "u"]
    }
    mylang.set_phonemes(phonemes=phonemes)

    # Set the parts of speech of the language
    parts_of_speech = ["noun", "verb", "adj", "prep", "det", "pron"]
    mylang.set_parts_of_speech(parts_of_speech=parts_of_speech)

    # Set the generation rules
    # Adj N (Prep Adj N) V Adj O (Prep Adj N)
    mylang.set_generation_rules({
        "S": [["sNP", "VP"], 1],  # Sentences generate subject NPs and VPs
        "VP": [["PosNegverb", "UnmarkedNP"], 0.7, ["PosNegverb"], 0.3],  # VPs generate verbs (and object NPs)
        "NP": [["det*nouny", "NOM"], 0.6, ["PRON"], 0.4],  # NPs can be a det NOM or a PRON
        "NOM": [["adj*nouny", "NOM"], 0.35, ["NoAdjNOM"], 0.65],  # NPs may take adjectives before the rest, recursive
        "NoAdjNOM": [["N", "PP*nom.__hash__"], 0.2, ["N"], 0.8],  # NoAdjNPs become nouns, or nouns with a PP
        "PP": [["prep*nouny", "UnmarkedNP"], 1]  # PPs always become prepositions followed by NPs
    })

    # Set independent probabilistic rules, e.g. pluralization, past tense
    mylang.set_unconditioned_rules({
        "sNP": [["UnmarkedNP"], "nom", 1],  # Subject NPs take the nominative
        "UnmarkedNP": [["NP"], "__hash__.nouny", 1],  # We want to make sure that words in the same NP can agree
        "PosNegverb": [["verb"], "pos", 0.9,  "neg", 0.1],  # Negative verbs take a prefix
        "N": [["noun"], "sg", 0.8, "pl", 0.2],  # Nouns may be singular or plural
        "PRON": [["pron"], "1st.sg", 0.2, "2nd.sg", 0.1, "3rd.sg", 0.25, "1st.pl", 0.15, "2nd.pl", 0.05, "3rd.pl", 0.25]
    })

    # Set the agreement rules
    mylang.set_agreement_rules({
        "verb": [["nom", "nouny"], [["sg", "pl"], ["1st", "2nd", "3rd"]]],  # Verbs agree with nominative nouns
        "det": [["noun", "__hash__"], [["sg", "pl"]]]  # Determiners agree with their head nouns
    })

    # Set a dictionary for the language
    mylang.set_dictionary({
        "verb": ["eat", "see", "find"],
        "noun": {
            "nonepenthetic.3rd": ["man", "woman", "teacher", "chair", "computer"],
            "epenthetic.3rd": ["box", "fox", "pox"]
        },
        "adj": ["big", "small"],
        "det": ["the"],
        "prep": ["on", "under"],
        "pron": [""]
    })

    # Set an inflection paradigm for determiners
    mylang.set_inflection_paradigms([
        ["verb", {
            ("sg", "1st"): "-",
            ("sg", "2nd"): "-",
            ("sg", "3rd"): "-s",
            ("pl"): "-",
        }],
        ["verb", {
            ("pos"): "-",
            ("neg"): "un-",
        }],
        ["noun", {
            "sg": "-",
            ("pl", "/x_"): "-es",
            ("pl", "/C_"): "-s",
        }],
        ["pron", {
            ("sg", "1st", "nom"): "-I",
            ("sg", "1st", "*nom"): "-me",
            ("sg", "3rd", "nom"): "-she",
            ("sg", "3rd", "*nom"): "-her",
            ("pl", "1st", "nom"): "-we",
            ("pl", "1st", "*nom"): "-us",
            "2nd": "-you",
            ("pl", "3rd", "nom"): "-they",
            ("pl", "3rd", "*nom"): "-them"
        }]
    ])

    # Save the language
    mylang.dump_language(os.path.join("Datasets", language_name) )

    # Make num_train and num_test integers
    num_train = int(num_train)
    # num_train should be a power of 10 and that it's at least 10 sentences
    assert math.log10(num_train) % 1 == 0 and num_train >= 10

    # We start by generating many sentences
    sentences, sequences = mylang.generate_sentences(num_sentences=num_train, required_words=None, sampling_method="uniform", regenerate_exception_sentences=True)
    # print(sentences[0])
    # print(sequences[0])
    # change the verb to be pst if it contains the property prs

    # Save these now
    for num_train_group in range(1, int(math.log10(num_train) ) + 1):
        language.save_sentences(sentences=sentences[:10 ** (num_train_group + 1)],
                                filepath=os.path.join("Datasets/",
                                                      language_name,
                                                      f"{10 ** (num_train_group + 1)}_train_sentences.txt") )


# =====================================GENERALLY HELPFUL METHODS========================================================
def generate_and_save_sentences(lang, language_name, num_sentences, sentence_prefix, required_words=None):
    # Create the directory if it does not exist
    directory_path = os.path.join("Languages", language_name)
    os.makedirs(directory_path, exist_ok=True)
    os.makedirs(os.path.join(directory_path, "train"), exist_ok=True)

    # Generate some sentences
    sentences, _ = lang.generate_sentences(num_sentences, required_words)
    rand.shuffle(sentences)

    # These are the sizes of the test
    training_sizes = np.logspace(1, 7, num=7, base=10, dtype=int)

    # Gradually increase the number of training sentences
    for training_set_index in range(len(training_sizes)):
        # The number of sentences we need is the training size, minus what's already in there
        # The number of sentences that's already in there is the previous training set index
        if training_set_index == 0:
            prev_num_training_examples = 0
        else:
            prev_num_training_examples = training_sizes[training_set_index - 1]
        # Get the number of sentences that we want to train the model on
        size_current_training_set = training_sizes[training_set_index] - prev_num_training_examples

        # Get that number of sentences uniquely
        training_set = sentences[prev_num_training_examples:size_current_training_set]

        # Save them
        language.save_sentences(sentences=training_set,
                                filepath=os.path.join(directory_path, "train",
                                                      f"{training_sizes[training_set_index]}_"
                                                      f"{sentence_prefix}_sentences.txt"))

    # Return the sentences you generate
    return sentences


if __name__ == "__main__":
    # Generate the frisian sentences
    generate_toy_data(num_train=10000)
