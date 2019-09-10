
import itertools
import collections

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances, pairwise_kernels

from soft_cosine import soft_cosine4, get_M
import utils


BDoc = collections.namedtuple('BDoc', ('ids', 'tokens', 'lemmas'))
NTDoc = collections.namedtuple('NTDoc', ('id', 'tokens', 'lemmas'))
Context = collections.namedtuple('Context', ('tokens', 'lemmas'))


def filter_stopwords(tokens, lemmas, stopwords):
    tokens_, lemmas_ = [], []
    for tok, lem in zip(tokens, lemmas):
        if lem.lower() not in stopwords:
            tokens_.append(tok)
            lemmas_.append(lem)
    return tokens_, lemmas_


def load_bernard(nwords, overlap, context_words=0, stopwords=None):
    docs, id2doc = [], {}

    for sermo, group in itertools.groupby(
            filter(None, utils.read_bernard_lines(drop_pc=True, lower=True)),
            key=lambda tup: utils.parse_id(tup[0])['sermo']):

        group = list(group)

        for idx in range(0, len(group), nwords - overlap):
            # stop when end is reached
            if idx + nwords == len(group):
                break

            ids, tokens, _, _, lemmas = zip(*group[idx: idx + nwords])
            # id
            for i in ids:
                id2doc[i] = len(docs)
            # get document
            if stopwords is not None:
                tokens, lemmas = filter_stopwords(tokens, lemmas, stopwords)
            doc = BDoc(ids, ' '.join(tokens), ' '.join(lemmas))
            # add context
            context = None
            if context_words > 0:
                words = (context_words - nwords) // 2
                start = max(0, idx - words)
                stop = min(len(group) - 1, idx + nwords + words)
                _, tokens, _, _, lemmas = zip(*group[start: stop])
                if stopwords is not None:
                    tokens, lemmas = filter_stopwords(tokens, lemmas, stopwords)
                context = Context(' '.join(tokens), ' '.join(lemmas))

            docs.append((doc, context))

    return docs, id2doc


def load_NT(context_words=0, stopwords=None):
    docs, id2doc = [], {}
    lines = sorted(utils.read_NT_lines(), key=lambda tup: tup[0])

    for book, group in itertools.groupby(lines, key=lambda tup: tup[0]):
        group = list(group)

        for idx, (book, chapter, verse_num, tokens, lemmas) in enumerate(group):
            # id
            id2doc[book, chapter, verse_num] = len(docs)
            # doc
            tokens, lemmas = tokens.split(), lemmas.split()
            if stopwords is not None:
                tokens, lemmas = filter_stopwords(tokens, lemmas, stopwords)
            doc = NTDoc((book, chapter, verse_num), ' '.join(tokens), ' '.join(lemmas))
            # context
            context = None
            if context_words > 0:
                words = (context_words - len(tokens)) // 2
                # left words
                left_tokens, left_lemmas = [], []
                idx_ = idx - 1
                while len(left_tokens) < words and idx_ >= 0:
                    *_, left_token, left_lemma = group[idx_]
                    left_tokens.extend(left_token.split()[::-1])
                    left_lemmas.extend(left_lemma.split()[::-1])
                    idx_ -= 1
                left_tokens.reverse()
                left_lemmas.reverse()
                # right words
                right_tokens, right_lemmas = [], []
                idx_ = idx + 1
                while len(right_tokens) < words and idx_ <= len(group) - 1:
                    *_, right_token, right_lemma = group[idx_]
                    right_tokens.extend(right_token.split())
                    right_lemmas.extend(right_lemma.split())
                    idx_ += 1

                context = Context(
                    ' '.join(left_tokens[-words:] + tokens + right_tokens[:words]),
                    ' '.join(left_lemmas[-words:] + lemmas + right_lemmas[:words]))

            docs.append((doc, context))

    return docs, id2doc


def load_refs(b_id2doc, nt_id2doc):
    missing = 0
    mappings = utils.read_mappings()
    for ref_type, (book_num, book, chapter, verse_num), span in utils.read_refs():
        b_docs = []

        for i in span:
            try:
                b_docs.append(b_id2doc[i.lower()])
            except KeyError:
                pass
        if not b_docs:
            missing += 1
            continue
        b_docs = list(set(b_docs))

        try:
            NT_book = mappings[book]
            if book_num is not None:
                NT_book = book_num + ' ' + NT_book
            nt_doc = nt_id2doc[NT_book, chapter, verse_num]
            yield ref_type, b_docs, nt_doc
        except Exception:
            missing += 1
    print(missing)


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
    bernard, b_id2doc = load_bernard(
        args.nwords, args.overlap, context_words=args.context_words, stopwords=stopwords)
    bernard, b_context = zip(*bernard)
    bernard = list(bernard)
    nt, nt_id2doc = load_NT(context_words=args.context_words, stopwords=stopwords)
    nt, nt_context = zip(*nt)
    nt = list(nt)
    refs = list(load_refs(b_id2doc, nt_id2doc))

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
