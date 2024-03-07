import json
import os
import numpy.random as nprand
import random
from copy import deepcopy

from tqdm import tqdm


# Method used to save sentences to a txt file
def save_sentences(sentences, filepath):
    # Open the file write only
    with open(filepath, "w") as file:
        # We want each string to be ended with a period and followed by a new line
        # We also want the last sentence to have a period though
        file.write(".\n".join(sentences) + ".")


# Helper method used for probabilistic CFGs
def choose_state(rule, is_generation):
    choices = []
    probabilities = []
    # If it's generation, use the format "A": [["B", "C*p", "..."], x, ["D"], y, ...]
    # If it's not generation it's unconditioned, use the format "A": ["a", "p1", x, "p2", y, ...]
    # Either way we alternate between the choices and the probabilities, the only difference is that unconditioned
    #   starts with the state of the output
    # Iterate through the values in the list of the rule
    for i in range(0 if is_generation else 1, len(rule), 2):
        # The first element is going to be a choice
        choices.append(rule[i])
        # The second element is going to be a probability
        probabilities.append(rule[i + 1])
    # Returns the chosen option and optionally, the next state
    return random.choices(choices, probabilities)[0], None if is_generation else rule[0]


# Load a Language object form a file
def load_language(directory_path):
    # Retrieve it from JSON format
    with open(os.path.join(directory_path, "mylang.json"), "r") as file:
        data = json.load(file)
    # Create an object with that data
    new_language = Language()
    new_language.phonemes = data["phonemes"]
    new_language.syllables = data["syllables"]
    new_language.syllable_lambda = data["syllable_lambda"]
    new_language.generation_rules = data["generation_rules"]
    new_language.unconditioned_rules = data["unconditioned_rules"]
    new_language.agreement_rules = data["agreement_rules"]
    new_language.words = data["words"]
    # We want to keep the set datatype, so we have to add this line
    new_language.word_set = set(data["word_set"])
    # We changed the paradigms to have strings as keys, so now we need to undo that change
    temporary_paradigms = []
    for paradigm in data["inflection_paradigms"]:
        # Add the paradigms back in the original format, with [pos, {feature tuple: affix}]
        temporary_paradigms.append([paradigm[0],
                                    {tuple(key.split(".")): value for key, value in paradigm[1].items()}])
    new_language.inflection_paradigms = temporary_paradigms
    return new_language


# Method used to inflect the agreed_lexeme_sequences according to some rules
# agreed_lexeme_sequences is a list of agreed_lexeme_sequences, paradigms is formatted as Language.inflection_paradigms
def inflect(agreed_lexeme_sequences, paradigms, phonemes):
    # Make sure that a list of lists is passed in
    assert type(agreed_lexeme_sequences[0]) is list
    inflected_sentences = []
    # For each agreed_lexeme_sequence, we want to run the agreement algorithm
    for agreed_lexeme_sequence in agreed_lexeme_sequences:
        inflected_words = []
        # Do agreement over each lexeme in the sequence
        for lexeme_with_agreement in agreed_lexeme_sequence:
            # Get the lexeme and the properties of the word
            lexeme, properties = lexeme_with_agreement
            # See if it agrees with any rules (can be 0, 1, or more)
            # A for loop is used here since it's possible that more than one rule applies
            for rule in paradigms:
                # If the pos of a rule isn't in the lexeme, we continue
                if rule[0] not in properties:
                    continue
                # Otherwise, the rule applies
                # We want to see the number of applicable inflections, since it should be equal to exactly 1
                applicable_inflections = []
                # rule_properties is a collection (or single) property that must apply for the inflection to be used
                # inflection is the inflection that will be applied (e.g. "-suf" or "pref-")
                for rule_properties, inflection in rule[1].items():
                    # We want to check that each property applies. If any don't we continue
                    perfect_property_match = True
                    # If the keys are tuples, then all properties must match
                    # If they're not, then only the single property must match
                    for rule_property in (rule_properties if type(rule_properties) == tuple else [rule_properties]):
                        # If the rule property doesn't apply, then it's not a perfect match
                        # This means we don't add the inflection to applicable inflections
                        # If a rule property is allophonic, we handle it differently
                        if rule_property[0] == "/":
                            # Get the part after the slash
                            environment = rule_property[1:]
                            # Get the environment before and after the underscore
                            left_environment, right_environment = environment.split("_")
                            # We don't want circumfixes or environments that rely on right and left phonemes
                            assert not (left_environment and right_environment)
                            assert inflection[0] == "-" or inflection[-1] == "-"
                            # If the inflection is "-suf", then we want to look at the phonemes to the left of the "-"
                            # An example environment is "/C_"
                            # If it's a suffix, it only has a left environment
                            # We will want to look at the sounds before where the "-" would attach
                            # The triggers are the sounds in the words
                            # The environment is from the inputted property
                            if left_environment:
                                environment = left_environment
                                triggers = lexeme[-len(environment.replace("*", "")):]
                            # If it's a prefix, we want to look at the sounds after the "-"
                            elif right_environment:
                                environment = right_environment
                                triggers = lexeme[:len(environment.replace("*", ""))]
                            else:
                                raise Exception("Unknown error! Neither left nor right environment.")
                            # The triggers should be the same length as the environment without the *
                            assert len(triggers) == len(environment.replace("*", ""))
                            # We want to make sure that every phoneme on the left matches the environment
                            num_asterisks = 0
                            # We match every trigger, and compare it with the one or two characters in environment
                            for i in range(len(triggers)):
                                # The action we take depends on if the environment's character is a natural class,
                                #   phoneme, or asterisk. If there is an asterisk, we negate
                                is_asterisk = False
                                # If it's an asterisk, we make sure the following character is negative
                                if environment[i] == "*":
                                    num_asterisks += 1
                                    is_asterisk = True
                                # The action we do depends on whether it's negated!
                                # A is whether the trigger is in the environment, or the trigger is the environment
                                # A = triggers[i] in phonemes[environment[i + num_asterisks]]  # (natural class)
                                # A = triggers[i] != environment[i + num_asterisks]  # (phoneme)
                                # We represent both of these with the variable match
                                # B is whether the environment has an asterisk
                                # B = is_asterisk
                                # If there is no asterisk, then the trigger not being in the natural class breaks
                                # B = false, A = false --> (execute and break) True
                                # If there is no asterisk, then the trigger being in the natural class continues
                                # B = false, A = true --> (do not execute) False
                                # If there is an asterisk, then the trigger not being in the natural class continues
                                # B = true, A = false --> (do not execute) False
                                # If there is an asterisk, then the trigger being in the natural class breaks
                                # B = true, A = true --> (execute and break) True
                                # Find the appropriate A
                                if environment[i + num_asterisks] in phonemes:
                                    match = triggers[i] in phonemes[environment[i + num_asterisks]]
                                else:
                                    match = triggers[i] == environment[i + num_asterisks]
                                # Now we do the logical operation described above
                                # We want to enter this statement if the triggers don't match the environmnet
                                if bool(match) == bool(is_asterisk):
                                    perfect_property_match = False
                                    break
                        # If a rule property requires a feature's absence, we handle it differently
                        elif rule_property[0] == "*":
                            # We just want to make sure that this feature does not exist in the lexeme's properties
                            if rule_property[1:] in properties:
                                perfect_property_match = False
                        # All other rule properties just require the property to exist in the lexeme's properties
                        else:
                            # This is the truth condition for non-allophonic properties to be true
                            if rule_property not in properties:
                                perfect_property_match = False
                        # If it's not a perfect_property_match, we break the loop
                        if not perfect_property_match:
                            break
                    # If it's not a perfect_property_match, we don't add this inflection to the applicable inflections
                    # Instead, we continue the loop
                    if not perfect_property_match:
                        continue
                    # If it's a perfect property match, we do indeed want to add it to the applicable inflections
                    applicable_inflections.append(inflection)
                # There should be exactly one key that works with the agreements of the lexeme
                if len(applicable_inflections) != 1:
                    raise Exception(f"Incorrect number of applicable inflections ({len(applicable_inflections)}) "
                                    f"for {lexeme_with_agreement} "
                                    f"given rule {rule}. \n"
                                    f"Debug info: properties = {properties}, "
                                    f"applicable_inflections = {applicable_inflections}")
                # We apply the affix to the lexeme
                affix = applicable_inflections[0]
                # The affixed form depends on the position of the dash.
                # For simple affixes, we just attach it where the dash it
                lexeme = affix.replace("-", lexeme)
            # Once we go through all the rules, we can add it to inflected words
            # We add them with the properties in case we want to look at them
            inflected_words.append([lexeme, properties])
        # We now take the inflected words and turn them into the surface form
        inflected_sentence = " ".join([word_and_properties[0] for word_and_properties in inflected_words])
        # Add this to the inflected sentences
        inflected_sentences.append(inflected_sentence)
    # Return the inflected sentences
    return inflected_sentences


# Language class used to generate words according to a distribution
class Language:
    # Constructor
    def __init__(self, language=None):
        # All the fields depend on whether language is passed in
        # Maps C and V to a list of phonemes
        self.phonemes = {} if language is None else deepcopy(language.phonemes)
        self.syllables = [] if language is None else deepcopy(language.syllables)
        self.syllable_lambda = 1 if language is None else deepcopy(language.syllable_lambda)
        self.generation_rules = {} if language is None else deepcopy(language.generation_rules)
        self.unconditioned_rules = {} if language is None else deepcopy(language.unconditioned_rules)
        self.agreement_rules = {} if language is None else deepcopy(language.agreement_rules)
        self.words = {} if language is None else deepcopy(language.words)
        self.word_set = set() if language is None else deepcopy(language.word_set)
        self.inflection_paradigms = [] if language is None else deepcopy(language.inflection_paradigms)

    # Set the phonemes
    def set_phonemes(self, phonemes):
        self.phonemes = phonemes

    # Set the syllables
    def set_syllables(self, syllables):
        self.syllables = syllables

    # Set the lambda for the number of syllables in the language. The number is automatically summed to 1.
    def set_syllable_lambda(self, syllable_lambda=1):
        self.syllable_lambda = syllable_lambda

    # Set part of speech
    # Not encoded in a separate variable
    def set_parts_of_speech(self, parts_of_speech):
        # The parts of speech passed in must be in the form of a list
        assert type(parts_of_speech) is list
        # For each part of speech, define the words belonging to that part of speech as an empty list
        for part_of_speech in parts_of_speech:
            self.words[part_of_speech] = []

    # Set sentence generation rules according to CFGs
    # The format of a rule is "A": [["B", "C*p", "..."], x, ["D"], y, ...]
    #   A is the input state
    #   B, C, ... are the outputs states
    #       C*p means that the property p will be removed from C
    #   x is the probability of the rule taking place
    #   D is another output state that takes place with probability y
    # An example is "S": [["sN", "VP"], 1]
    #   This means sentence goes to subject noun and verb phrase with probability 1
    # Another example is "VP": [["verb", "oN"], 0.7, ["verb"], 0.3]
    #   This means verb phrases take an object noun with probability 0.7
    # The probabilities must sum to 1, but this isn't checked
    def set_generation_rules(self, generation_rules):
        self.generation_rules.update(generation_rules)

    # Sets sentence generation rules for individual words that are not conditioned by other words
    # The format of a rule is "A": [["a"], "p1", x, "p2", y, ...]
    #   A is the input state
    #   a is the output state, it must be wrapped in a list
    #   x is the probability of a having property p1, y is the probability of having property p2
    # An example is "sN": [["noun"], "sing", 0.8, "pl", 0.2]
    #   This means that subject nouns map to nouns, with the feature singular with probability 0.8 and plural with 0.2
    def set_unconditioned_rules(self, unconditioned_rules):
        self.unconditioned_rules.update(unconditioned_rules)

    # Sets the agreement rules for words with a property or terminal
    # The format of a rule is "t": [["p1", "p2", ...], [["q1", "q2", ...], ["r1", ...], ...]]
    #   t is the property or terminal that takes the agreement affixes
    #   p1, p2, ... are the agreement properties that a word that t agrees with must have to trigger agreement
    #       There must be exactly one word with ALL the properties that are needed to trigger agreement
    #       Otherwise, an error is raised
    #   ["q1", "q2", ...] is a set of properties that determine feature q
    #       The word that satisfies the agreement properties must have exactly one of the features in q
    #       Otherwise, an error is raised
    #   ["r1", ...], ... are other sets of properties that determine other features
    # An example is "verb": [["nom", "noun"], [["sg", "pl"], ["1st", "2nd", "3rd"]]]
    #   This means that verbs must agree with a word that has the property nominative and noun
    #   Then, the verb agrees with that word, taking either the property singular or plural, and 1st, 2nd, or 3rd
    # The agreement rules don't dictate the inflections, just what words agree with what
    # FOR NOW, WORDS CAN ONLY AGREE WITH ONE OTHER WORD. POTENTIALLY CHANGE THIS LATER.
    def set_agreement_rules(self, agreement_rules):
        self.agreement_rules.update(agreement_rules)

    # Allows you to pass in a custom vocabulary
    # The vocabulary is a dict of the following format:
    # {
    #     "pos1": [
    #         "w1", "w2", "w3", ...
    #     ],
    #     "pos2": {
    #         "p1": [
    #             "w4", "w5", "w6", ...
    #         ],
    #         "p2": [
    #             "w7", "w8", "w9", ...
    #         ]
    #     }
    #     "pos3": [...],
    #     "pos4": [...],
    #     ...
    # }
    # "pos1", "pos2", ... are the parts of speech that represent the terminal states in the PCFG
    # "p1", "p2", ... are properties intrinsic to any word that's "pos2"
    # "w1", "w2", ... are words in their base form
    # Every dictionary that exists as the value of a pos key must only have list values, you can use periods
    #   to separate multiple features. Assigning a pos key to a list is an abbreviation for creating a single paradigm
    #   called f'{pos}_main'
    # FOR NOW, IF A WORD IS MULTIPLE PARTS OF SPEECH IT WILL ONLY KEEP THE FIRST
    def set_dictionary(self, imported_dictionary):
        # Iterate through the words in the dictionary and add them to the lexicon
        for pos, lexemes_or_classes in imported_dictionary.items():
            # If it's just a list, then we add every word to the lexicon
            if type(lexemes_or_classes) is list:
                # Add each of the words to the lexicon
                for lexeme in lexemes_or_classes:
                    self.add_word(lexeme, pos, f'{pos}_main')
            # Otherwise, we also need to add the features
            elif type(lexemes_or_classes) is dict:
                # We do the same for each class in the pos
                for word_class in lexemes_or_classes:
                    # Add each of the words to the lexicon
                    for lexeme in lexemes_or_classes[word_class]:
                        self.add_word(lexeme, pos, word_class)
            # We shouldn't get here
            else:
                raise Exception("Values for parts of speech not lists or dictionaries")

    # Define an inflection pattern for a given paradigm
    # The format of a rule is ["w", {("p1", "p2", ...): "-s1", ("q1", ...): "-s2", ...}]
    #   w is the property of the word that triggers a check for whether this word inflects for this rule
    #   ("p1", "p2", ...) is a tuple of properties that result in inflection being triggered
    #       There must be exactly one tuple that triggers inflection for a given word
    #       Otherwise, an error is raised
    #   -s1 is a suffix that is appended to the word if the properties that trigger inflection are met
    #       Prefixes (pref-) and circumfixes (cir-cum) are also supported
    #   ("q1", ...): "-s2" is another tuple of properties that triggers inflection with suffix "-s2"
    # If we call ["w", {("p1", "p2", ...): "-s1", ("q1", ...): "-s2", ...}] R1, the format of the input is [R1, R2, ...]
    #   The rules apply in order
    #   If you're adding a single inflection paradigm, wrap the rule in a list
    # An example for one rule is ["noun", {"sg": "-", "pl": "-ol"}]
    #   For this rule, we see that nouns must agree with either singular of plural. If they agree with none or both,
    #       an error is thrown
    #   Whichever feature it agrees with results in that suffix getting added. The singular is unmarked while the
    #       plural takes the suffix "-ol"
    def set_inflection_paradigms(self, inflection_paradigms):
        # The inflection_paradigm is a dictionary. All inflections are suffixes
        self.inflection_paradigms.extend(inflection_paradigms)

    # Add words to our lexicon at the end of the list for that part of speech, but control for the surface form
    def add_word(self, surface_form, part_of_speech, paradigm):
        # Add the word to the language's word
        self.words[part_of_speech].append((surface_form, paradigm))
        # Add the word to the language's word_set
        self.word_set.add(surface_form)

    # Add words to our lexicon at the end of the list for that part of speech
    # Words can be individual words or tuples consisting of (word, paradigm_number)
    # Paradigm is a mandatory string parameter which defines the words as being part of that paradigm
    # Returns a list containing the new words
    # If you want to include more than one property in the paradigm, separate them with periods
    # New words are guaranteed to be not in wordset
    def generate_words(self, num_words, part_of_speech, paradigm, add_to_lexicon=True):
        # Generate words with each phoneme in a given class appearing with the same frequency
        # All syllable types appear with equal frequency too
        # We use a while loop since we don't want duplicate words (for now!)
        new_words = []
        while len(new_words) < num_words:
            # Select a random number of syllables (+1 for non-empty syllables)
            num_syllables = nprand.poisson(self.syllable_lambda) + 1
            # For every syllable, choose a random syllable structure and construct a word out of it
            word = ""
            for syllable in range(num_syllables):
                syllable_structure = random.choice(self.syllables)
                # For every natural class in the syllable, choose a random phoneme that fits that description
                for natural_class in syllable_structure:
                    # Find a random phoneme from that natural class and add it to the word
                    word += (nprand.choice(self.phonemes[natural_class]))
            # If we generated a new word, we add it to our lexicon and to the words we made
            if word not in self.word_set:
                new_words.append((word, paradigm))
                # We only add a new word to this language if add_to_lexicon is True
                if add_to_lexicon:
                    self.word_set.add(word)
        # Add the new_words to the part of speech they were made for, if add_to_lexicon is True
        if add_to_lexicon:
            self.words[part_of_speech] += new_words
        # Now return the list in case it's needed
        return new_words

    # Generate sentences according to a certain distribution
    # Required words is by default None.
    #   If you want to generate sentences with words from a specific set, you pass in a dictionary.
    #   This dictionary maps pos of words to a list tuples of words and paradigms
    #   For example, required_words = {pos: []}
    # If you're generating sentences with required_words, note that all parts of speech not in required_words will
    #   be drawn with Zipf's distribution as normal. This may mean that if a sentence is generated with no terminal
    #   pos in required words, then there won't be any words from required words in the sentence, and that if a
    #   sentence has more than one terminal pos in required words, all of those will be drawn from required words.
    #   All words drawn from required_words are drawn uniformly.
    # The default sampling method is Zipfian, set with 'zipfian' for sampling_method. You may also set this as uniform,
    #   setting sampling_method to 'uniform'. All other values will raise an error.
    def generate_sentences(self, num_sentences, required_words=None, sampling_method='zipfian',
                           regenerate_exception_sentences=False):
        # Make sure that sampling_method is 'zipfian' or 'uniform'
        if sampling_method not in ['zipfian', 'uniform']:
            raise ValueError(f'Sampling method {sampling_method} illegal.')
        # Make sure that num_sentences is a strictly positive integer
        assert type(num_sentences) is int and num_sentences > 0

        # Prepare the sentences we want
        sentences = []
        agreed_lexeme_sequences = []
        # We also want to keep track of the number of exception sentences
        exception_sentences = 0
        for _ in tqdm(range(num_sentences)):
            # If the code raises an exception, we stop if regenerate_exception_sentences is true
            try:
                # GENERATE THE TERMINAL POS STATES AND PROPERTIES
                # We want the sentence to only contain terminal nodes, but we start with the start state
                sentence = "S-"
                # All terminals and properties are all lowercase.
                # If the sentence contains uppercase letters, then we have not finished constructing it
                while sentence.lower() != sentence:
                    # Continually replace non-terminal nodes until the sentence doesn't change
                    temp_sentence = ""
                    for state in sentence.strip().split(" "):
                        # Split this state from any properties it has
                        raw_state, existing_properties = state.split("-")
                        # Separate the existing properties into a list
                        # The separated_existing_properties list will contain multiple elements if there is a "." in there
                        # If there are no existing properties, then return an empty list
                        separated_existing_properties = existing_properties.split(".")
                        # We don't want a zero string property, so if the only element in the list is a zero string,
                        #   then we make separated_existing_properties an empty list
                        if separated_existing_properties == [""]:
                            separated_existing_properties = []
                        # If the raw states is a terminal part of speech, continue the loop
                        if raw_state in self.words.keys():
                            # We want to add the whole state
                            temp_sentence += state + " "
                            continue
                        # If it's a generation rule, then we add words individually to the temp_sentence
                        if raw_state in self.generation_rules:
                            # Choose the next state(s) for this state
                            # There will never be a new_property from a generation_rule
                            next_states, new_property = choose_state(rule=self.generation_rules[raw_state],
                                                                     is_generation=True)
                            # For each next state, add the new property, but remove the property after *
                            for next_state in next_states:
                                # See if there is something to remove, if there isn't then there are no unwanted properties
                                # If this throws an error, it means that there was more than 1 "*" in the next_state
                                # If there is one unwanted property, then we split it on the asterisk
                                if "*" in next_state:
                                    true_next_state, unwanted_properties = next_state.split("*")
                                # If there are no wanted properties, the next_state is the true_next_state
                                else:
                                    true_next_state, unwanted_properties = next_state, "__no_unwanted__"
                                # Remove the property_to_be_removed for this next_state only, if applicable
                                updated_existing_properties = deepcopy(separated_existing_properties)
                                # For every unwanted property
                                for unwanted_property in unwanted_properties.split("."):
                                    # If the unwanted_property is in the existing_property, then we kick it
                                    if unwanted_property in updated_existing_properties:
                                        updated_existing_properties.remove(unwanted_property)
                                    # If the unwanted property is a HASH, remove the whole hash value
                                    if unwanted_property == "__hash__":
                                        updated_existing_properties = [
                                            prop for prop in updated_existing_properties if "__hash__" not in prop
                                        ]
                                # Add this next state, potentially without the unwanted_property, to the temp_sentence
                                temp_sentence += (f'{true_next_state}-'
                                                  f'{".".join(updated_existing_properties)} ')
                        # If it's an unconditioned rule, we see if we want to add any properties
                        elif raw_state in self.unconditioned_rules:
                            # Choose the property for this state
                            # There will only ever be one new next_state, but it will be wrapped in a list
                            # There may be more than one new property though. These are period separated
                            new_properties, next_states = choose_state(rule=self.unconditioned_rules[raw_state],
                                                                       is_generation=False)
                            for new_property in new_properties.split("."):
                                # If the new_property is "__hash__", we want to add a hash value as the new property
                                if new_property == "__hash__":
                                    # We create a pseudorandom number and assign it to the next state of this object
                                    # Remove the "0." though since that would mess everything up
                                    # Some values will have "-" in the form of e- something, we want to remove that too
                                    separated_existing_properties.append(
                                        f"__hash__:{str(random.random())[2:].replace('-', '')}"
                                    )
                                # If the new property is not hash, then we just add it to the list of existing properties
                                else:
                                    # Add the new property to the existing properties
                                    separated_existing_properties.append(new_property)
                            # Add the new next_state with the new property to the temp_sentence
                            temp_sentence += (f'{next_states[0]}-'
                                              f'{".".join(separated_existing_properties)} ')
                        # Sanity check: if we enter this loop, the state should be in either generation or unconditioned
                        else:
                            raise Exception(f"Invalid state for input {state} given raw_state {raw_state} \n" +
                                            f"Make sure this is a key in generation or unconditioned rules.")
                    # Update the value of sentence
                    sentence = temp_sentence

                # REPLACE THE TERMINAL POSs WITH WORDS GENERATED ACCORDING TO THE ZIPFIAN DISTRIBUTIAN
                # Start with splitting the word into its pieces again
                preagreement_words = []
                for preagreement_word in sentence.strip().split(" "):
                    # Split each word into the lexeme and its properties
                    terminal, properties = preagreement_word.split("-")
                    # Add the lexeme and a list of properties to the preagreement words
                    preagreement_words.append([terminal, properties.split(".")])
                preagreement_lexemes = []
                for preagreement_word in preagreement_words:
                    # Get the terminal part of speech (pos) and the properties of the word
                    pos, properties = preagreement_word
                    # Generate a word according to Zipf's distribution
                    # If there are no word which we are required to use, then we're good!
                    # If there are required words but the part of speech is not in required words, we get a word according
                    #   to the distribution we set earlier
                    if required_words is None or pos not in required_words:
                        # At this point we've checked and know that sampling_method is a valid choice
                        # Draw a word randomly according to the distribution we selected
                        if sampling_method == 'zipfian':
                            # Set the skew parameter for Zipf's distribution
                            skew = 1.2  # Note: This parameter can be changed. Find a naturalistic one
                            # There isn't a nice way to generate an index according to Zipf's law
                            # The way we do it here is we generate a random index according to an unbounded distribution
                            # If it is outside the range of our list, we generate another one
                            # Otherwise, we use it
                            index = -1
                            while index == -1:
                                # We generate an index, subtracting 1 since Zipf's starts from 1
                                index = nprand.zipf(skew, 1)[0] - 1
                                # If it's out of the range, we reset the index to 0
                                if index >= len(self.words[pos]):
                                    index = -1
                            # If it is in the range, we exit our loop and get the word and paradigm
                            word, paradigm = self.words[pos][index]
                        # Draw a word uniformly
                        elif sampling_method == 'uniform':
                            word, paradigm = random.choice(self.words[pos])
                    # If we want to generate words from a list of words, then we draw uniformly from that set
                    else:
                        # Get the words at random from the list
                        word, paradigm = random.choice(required_words[pos])
                    # Add the sentence to the word_sentence
                    # We also make the part of speech and the existing paradigm a new feature
                    # We use paradigm.split(".") since if an entry has more than one property we mark them with . boundaries
                    preagreement_lexemes.append([word, properties + [pos] + paradigm.split(".")])

                # ADD AGREEMENT PROPERTIES, NOT YET INFLECTING
                # Now we iterate over every word to see if it must agree with any other words
                # All the preagreement lexemes are stored as [word, properties]
                agreed_words = []
                # print(preagreement_lexemes)
                for preagreement_word in preagreement_lexemes:
                    # Check to see if there's a rule describing this word.
                    # If there isn't, our work is done, so we add it to agreed_words and continue
                    # If none of the properties of a word are in the agreement rules
                    agreement_properties = list(set(preagreement_word[1]) & set(self.agreement_rules))
                    if len(agreement_properties) == 0:
                        agreed_words.append(preagreement_word)
                        continue
                    # For now, we can only handle one agreement. We might change this later
                    assert len(agreement_properties) == 1
                    # Get the rule for that terminal
                    rule = self.agreement_rules[agreement_properties[0]]
                    required_properties = set(rule[0])
                    # If there is, we want to find THE word that triggers agreement
                    words_triggering_agreement = []
                    for other_word in preagreement_lexemes:
                        # Make sure it has all the right properties
                        # We do this by making sure that every required property is in other word's properties
                        # For __hash__, we have to make sure that the hashes are the same for the agreement
                        # E.g. det.:123 must agree with noun.:123.sg, but the way that's phrase is:
                        #   "det": [["noun", "__hash__"], ["sg", "pl"]]
                        # This is problematic since the noun isn't marked with "__hash__" as a property
                        # What we need to do is:
                        # for each property in the other word's properties
                        #   if it's a hash property
                        #       # Make sure the hashes are the same
                        #   otherwise, check to see if it's in the required properties
                        #   if the property (hash or otherwise) is not found, then we continue. This word doesn't agree
                        #   otherwise, we add it to the words triggering agreement. There ultimately should only be one
                        # We first start by assuming that the word has all the required properties
                        all_required_properties_met = True
                        # We iterate over all the properties in the rule to ensure they are in the other word:
                        for required_property in required_properties:
                            # If it's a hash property, we handle it differently
                            if required_property == "__hash__":
                                # We want to make sure that the hash values are the same
                                # Firstly, find all the hash-like properties in this word and the other word
                                this_word_hash = [prop for prop in preagreement_word[1] if ":" in prop]
                                other_word_hash = [prop for prop in other_word[1] if ":" in prop]
                                # Every hash in this word must also be in the other word
                                # The reason this is the same (and not they need to full overlap) is as follows:
                                # - This word and other word come from the same state
                                # - Other word is the head, and may take on other hashes
                                # - This word won't take on other hashes (e.g. NP --> det NOM)
                                # Therefore, we just need to make sure that the overlap is the same as this word's
                                if set(this_word_hash) & set(other_word_hash) != set(this_word_hash):
                                    all_required_properties_met = False
                            # If it's a normal property, we just want to make sure it's in the other word's property list
                            else:
                                # If the required properties are missing, then we set all_required_properties_met to false
                                if required_property not in other_word[1]:
                                    all_required_properties_met = False
                            # If any property is not found, this isn't the word for us
                            if not all_required_properties_met:
                                break
                        # If any property is not found, this isn't the word for us:
                        if not all_required_properties_met:
                            continue
                        # If it does have all the right properties, add it to the list of words triggering agreement
                        words_triggering_agreement.append(other_word)
                    # Now we make sure there's EXACTLY ONE word triggering agreement
                    if len(words_triggering_agreement) != 1:
                        raise Exception(f"{len(words_triggering_agreement)} words triggered agreement for "
                                        f"{preagreement_word}. These words are {words_triggering_agreement}. "
                                        f"The rule that triggered it is {rule}. "
                                        f"Check rules.")
                    # Perfect! Now that we found the word that agrees with our preagreement word, we find the properties
                    # We want to check that the word triggering agreement has one of each property in each set
                    word_triggering_agreement = words_triggering_agreement[0]
                    # For every property that our preagreement word seeks, we look to see if trigger word has it
                    new_properties = []
                    # For every feature in that the preagreement word looks for
                    for sought_feature in rule[1]:
                        # Make sure that the intersection of the sought properties with the word triggering agreement is one
                        property_intersection = list(set(sought_feature) & set(word_triggering_agreement[1]))
                        # If there isn't exactly 1, then raise an error
                        if len(property_intersection) != 1:
                            raise Exception(f"Incorrect number of properties found for {preagreement_word}. \n"
                                            f"Sought feature: {sought_feature}. \n"
                                            f"Word triggering agreement: {word_triggering_agreement}")
                        # If there is exactly one feature, then we add it to the new properties of the preagreement word
                        new_properties.append(property_intersection[0])
                    # Now we have found all the new properties of the preagreement word!
                    # All that is left is to join the old and new properties, and add this word to our agreed word list
                    agreed_words.append([preagreement_word[0], preagreement_word[1] + new_properties])
                agreed_lexeme_sequences.append(agreed_words)

                # MAKE EACH WORD HAVE INFLECTIONS
                # We get only the first element of the output since we are only making one sentence
                inflected_words = inflect([agreed_words], self.inflection_paradigms, self.phonemes)[0]

                # FINALLY, GIVE THE SURFACE FORM
                # We only add the final sentence, not the properties, but we keep them along until the end for debugging
                sentences.append(inflected_words)
            # We always catch exceptions
            except Exception:
                # If we want to regenerate, then we keep track of the number of sentences we regenerated
                if regenerate_exception_sentences:
                    exception_sentences += 1
                # Otherwise, we raise an error
                else:
                    raise Exception('Error raised during sentence generation. Solve above.')
        # When we finish generating the number of sentences we want, then we print the number of regenerations if wanted
        if regenerate_exception_sentences:
            print(f"When generating {num_sentences}, {exception_sentences} were regenerated.")
        # Finally, we exit the loop and return the list of sentences
        return sentences, agreed_lexeme_sequences

    # Save the language in a given file
    def dump_language(self, directory_path):
        # Make the path to the file, if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        # Store it in JSON format
        with open(os.path.join(directory_path, "mylang.json"), 'w') as file:
            # Pretty much all data can be stored as is
            data = {
                "phonemes": self.phonemes,
                "syllables": self.syllables,
                "syllable_lambda": self.syllable_lambda,
                "generation_rules": self.generation_rules,
                "unconditioned_rules": self.unconditioned_rules,
                "agreement_rules": self.agreement_rules,
                "words": self.words,
                # Word_set must first be converted to a list
                "word_set": list(sorted(self.word_set))
            }
            # Inflection paradigms has tuples as keys.
            # We want to replace each tuple with a string with dots separating the properties
            temporary_paradigms = []
            for paradigm in self.inflection_paradigms:
                # The first value of every rule should be the same
                # The second value is the same as well, except each key is now a string with periods between properties
                # We only join the different parts with . if the key is a tuple
                temporary_paradigms.append([paradigm[0], {(".".join(key) if type(key) is tuple else key): value
                                                          for key, value in paradigm[1].items()}])
            # Don't forget to add it to the datafile!
            data["inflection_paradigms"] = temporary_paradigms
            # Dump the datafile
            json.dump(data, file)
