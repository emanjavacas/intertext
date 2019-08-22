
import itertools
import collections

import utils

BDoc = collections.namedtuple('BDoc', ('ids', 'tokens', 'lemmas'))
NTDoc = collections.namedtuple('NTDoc', ('id', 'tokens', 'lemmas'))


def load_bernard(nwords, overlap):
    docs = []
    id2doc = {}
    for sermo, group in itertools.groupby(
            filter(None, utils.read_bernard_lines(drop_pc=True, lower=True)),
            key=lambda tup: utils.parse_id(tup[0])['sermo']):
        group = list(group)
        for i in range(0, len(group), nwords - overlap):
            ids, tokens, _, _, lemmas = zip(*group[i: i+nwords])
            for i in ids:
                id2doc[i] = len(docs)
            docs.append(BDoc(ids, ' '.join(tokens), ' '.join(lemmas)))
    return docs, id2doc


def load_NT():
    docs = []
    id2doc = {}
    lines = sorted(utils.read_NT_lines(), key=lambda tup: tup[0])
    for book, group in itertools.groupby(lines, key=lambda tup: tup[0]):
        for book, chapter, verse_num, verse, lemma in group:
            id2doc[book, chapter, verse_num] = len(docs)
            docs.append(NTDoc((book, chapter, verse_num), verse, lemma))
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


nwords = 15
overlap = 7
bernard, b_id2doc = load_bernard(nwords, overlap)
nt, nt_id2doc = load_NT()
refs = list(load_refs(b_id2doc, nt_id2doc))

original_vocab = set()
for doc in nt + bernard:
    original_vocab.update(doc.lemmas.split())

W, vocab = utils.load_embeddings(original_vocab, 'embeddings.vec')
for w in original_vocab:
    if w not in vocab:
        vocab.append(w)
vocab = {w: idx for idx, w in enumerate(vocab)}

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(vocabulary=vocab).fit(doc.lemmas for doc in bernard + nt)
src_embs = tfidf.transform(doc.lemmas for doc in bernard).toarray()
trg_embs = tfidf.transform(doc.lemmas for doc in nt).toarray()
