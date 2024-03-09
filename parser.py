import nltk
import sys


# Context free grammar rules.

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# TODO: the following non terminals could be more succinct
NONTERMINALS = """
S -> NP V
S -> NP V NP
S -> NP P NP
S -> N V NP P N
S -> NP V Det Adj N
S -> N V P Det Adj N Conj N V
S -> NP V Adv Adj N
S -> N V P NP
S -> N V Adv Conj V NP
S -> N V P NP P NP
S -> N Adv V Det N Conj N V P NP Adv
S -> N V Det Adj N P NP Conj V N P Det Adj N
S -> N V Det Adj Adj Adj N P Det N P NP
NP -> N | Det N
"""


grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence: str) -> list[str]:
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)

    # Convert to lowercase
    words = [word.lower() for word in words]

    # Remove any word that does not contain at least one alphabetic character
    words = [word for word in words if any(c.isalpha() for c in word)]

    return words
    
def np_chunk(input_tree: nltk.tree.Tree) -> list[nltk.tree.Tree]:
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    noun_phrases = []
    for tree in input_tree:
        if tree.label() == "NP":
            if not any(subtree.label() == "NP" for subtree in tree):
                noun_phrases.append(tree)
            else:
                noun_phrases.extend(np_chunk(subtree) for subtree in tree)
        elif tree.label() == "N":
            noun_phrases.append(tree)
    return noun_phrases


if __name__ == "__main__":
    main()
