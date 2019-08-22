
import numpy as np

import tqdm


def soft_cosine4(ss, ts, M):
    """
    ss : n query docs in BOW format, np.array(n, vocab)
    ts : m indexed docs in BOW format, np.array(m, vocab)
    M : similarity matrix, np.array(vocab, vocab)

    returns : sims, soft cosine similarities, np.array(n, m)
    """
    sims = np.zeros((len(ss), len(ts)))
    MtsT = M @ ts.T
    den2 = np.sqrt(np.diag(ts @ MtsT))
    for idx, s in tqdm.tqdm(enumerate(ss), total=len(ss)):
        num = s[None, :] @ MtsT
        den1 = np.sqrt(s @ M @ s)
        sims[idx] = (num / ((np.ones(len(den2)) * den1) * den2))[0]
    return np.nan_to_num(sims, copy=False)


def get_M(S, vocab, beta=1):
    """
    Transform an input similarity matrix for a possibly reduced vocabulary space
    into the similarity matrix for the whole space (i.e. include OOVs). It assumes
    that OOVs are indexed at the end of the space - e.g. for a vocab of 1001 where
    the word "aardvark" doesn't have an entry in S (1000 x 1000), the entry for
    "aardvark" is 1001. By default the similarity vector for OOVs is a one-hot vector
    implying that the word is only similar to itself.

    S : input similarity matrix (e.g. sklearn.metrics.pairwise.cosine_similarity(W)
        where W is your embedding matrix), np.array(vocab, vocab)
    vocab : list of all words in your space
    beta : raise your similarities to this power to reduce model confidence on word
        similarities
    """
    M = np.zeros([len(vocab), len(vocab)])
    M[:len(S), :len(S)] = np.power(np.clip(S, a_min=0, a_max=np.max(S)), beta)
    np.fill_diagonal(M, 1)
    return M
