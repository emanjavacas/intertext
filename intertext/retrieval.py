

import collections

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_kernels

from soft_cosine import soft_cosine4, get_M

from intertext import utils
from intertext import data


def get_rankings(D, refs, ftype='sim'):
    D_ = D.argsort()
    if ftype == 'sim':
        D_ = D_[:, ::-1]
    rankings = []
    for _, src, trg in refs:
        src_ranks = [np.where(D_[s] == trg)[0][0] for s in src]
        for s in src:
            assert trg in set(D_[s].tolist()), "{}:{} not in array".format(s, trg)
        assert len(src_ranks) > 0, "Couldn't find id {}".format(trg)
        rankings.append(src_ranks)
    return rankings


def mrr(rankings, scoring='strict'):
    mrr = 0
    for ranks in rankings:
        # add one since ranks should start at 1
        mrr += 1 / (1 + (min(ranks) if scoring != 'strict' else max(ranks)))
    return mrr / len(rankings)


def acc_at(rankings, at, scoring='strict'):
    """
    - scoring : strict, take average hit
                single, just one paragraph counts
    """
    acc = 0
    for ranks in rankings:
        assert len(ranks) > 0, ranks
        if scoring == 'strict':
            acc += sum(rank < at for rank in ranks) / len(ranks)
        else:
            acc += any(rank < at for rank in ranks)
    return acc / len(rankings)


def acc_report(D, refs, ats=(5, 10, 20), **kwargs):
    d = collections.defaultdict(list)
    for ref, src, trg in refs:
        d[ref].append((ref, src, trg))
    output = collections.defaultdict(dict)
    for ref in d:
        rankings = get_rankings(D, d[ref], **kwargs)
        for at in ats:
            for scoring in ['strict', 'single']:
                output[ref][at, scoring] = acc_at(rankings, at, scoring=scoring)
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nwords', type=int, default=15)
    parser.add_argument('--overlap', type=int, default=7)
    parser.add_argument('--context_words', type=int, default=350)
    parser.add_argument('--lemmas', action='store_true')
    parser.add_argument('--embeddings_path')
    parser.add_argument('--lsi_dims', type=int, default=200)
    parser.add_argument('--stopwords')
    args = parser.parse_args()

    field = 'lemmas' if args.lemmas else 'tokens'
    stopwords = utils.read_stopwords(args.stopwords) if args.stopwords else None
    bernard, b_id2doc = data.load_bernard(
        args.nwords, args.overlap, context_words=args.context_words, stopwords=stopwords)
    bernard, b_context = zip(*bernard)
    bernard = list(bernard)
    nt, nt_id2doc = data.load_NT(context_words=args.context_words, stopwords=stopwords)
    nt, nt_context = zip(*nt)
    nt = list(nt)
    refs = list(data.load_refs(b_id2doc, nt_id2doc))

    # remove stopwords and compute vocabulary
    original_vocab = set()
    for doc in nt + bernard:
        original_vocab.update(getattr(doc, field).split())

    # # cosine
    tfidf = TfidfVectorizer(vocabulary=original_vocab).fit(
        getattr(doc, field) for doc in bernard + nt)
    src_embs = tfidf.transform(getattr(doc, field) for doc in bernard).toarray()
    trg_embs = tfidf.transform(getattr(doc, field) for doc in nt).toarray()
    D = pairwise_kernels(src_embs, trg_embs, metric='cosine', n_jobs=-1)
    np.save('cos_D.npy', D)

    # # soft cosine
    W, vocab = utils.load_embeddings(original_vocab, args.embeddings_path)
    for w in original_vocab:
        if w not in vocab:
            vocab.append(w)
    vocab = {w: idx for idx, w in enumerate(vocab)}
    tfidf = TfidfVectorizer(vocabulary=vocab).fit(
        getattr(doc, field) for doc in bernard + nt)
    src_embs = tfidf.transform(getattr(doc, field) for doc in bernard).toarray()
    trg_embs = tfidf.transform(getattr(doc, field) for doc in nt).toarray()
    S = pairwise_kernels(W, metric='cosine', n_jobs=-1)
    D = soft_cosine4(src_embs, trg_embs, get_M(S, vocab, beta=5))
    np.save('soft_D.npy', D)
