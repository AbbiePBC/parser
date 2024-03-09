import nltk
from parser import np_chunk

def test_np_chunk():

    sentence = """(S
    (N he)
    (V sat)
    (P in)
    (NP (Det the) (N armchair))
    (P in)
    (NP (Det the) (N home)))
    """
    tree = nltk.tree.Tree.fromstring(sentence)
    np_chunks = np_chunk(tree)
    assert len(np_chunks) == 2
    assert np_chunks[0].leaves() == ['the', 'armchair']
    assert np_chunks[1].leaves() == ['the', 'home']


