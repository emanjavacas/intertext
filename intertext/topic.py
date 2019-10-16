
import string
import glob
import collections
import itertools

import tqdm
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from intertext import utils


def load_bernard_documents():
    sermos = []
    for sermo, group in itertools.groupby(
            filter(None, utils.read_bernard_lines(drop_pc=True)),
            key=lambda tup: utils.parse_id(tup[0])['sermo']):
        sermo_data = []
        for idx, word, _, _, lemma in group:
            sermo_data.append((idx, word, lemma))
        sermos.append(sermo_data)

    return sermos


def load_NT_documents():
    docs = []
    lines = sorted(utils.read_NT_lines(), key=lambda tup: tup[0])
    for book, group in itertools.groupby(lines, key=lambda tup: tup[0]):
        doc = []
        for book, chapter, verse_num, verse, lemma in group:
            for word, lemma in zip(verse.split(), lemma.split()):
                doc.append(((book, chapter, verse_num), word, lemma))
        docs.append(doc)

    return docs


def load_patrologia(path='patrologia/processed-tokens/', punct=False, digits=False):
    docs = []
    p_ = str.maketrans('', '', string.punctuation)
    d_ = str.maketrans('', '', string.digits)
    for idx, f in enumerate(glob.glob(path + '*/*')):
        _, _, dirname, fname = f.split('/')

        with open(f) as f:
            doc = []
            for line in f:
                line = line.strip()
                if not punct:
                    line = line.translate(p_)
                if not digits:
                    line = line.translate(d_)
                line = line.split()
                if not line:
                    continue
                doc.extend(line)

        if not doc:
            continue

        docs.append(((dirname, fname), doc))

    return docs


def get_most_frequent():
    sermos = load_bernard_documents()
    docs = load_NT_documents()
    stopwords = collections.Counter()
    for doc in docs:
        _, w, l = zip(*doc)
        stopwords.update(l)
    for doc in sermos:
        _, w, l = zip(*doc)
        stopwords.update(l)
    return stopwords


def get_chunks(words, chunk_size, min_chunk_size=None):
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i+chunk_size]
        if min_chunk_size is None or len(chunk) > min_chunk_size:
            yield chunk


def chunk_collection(docs, chunk_size, **kwargs):
    return [chunk for doc in docs for chunk in get_chunks(doc, chunk_size, **kwargs)]


def get_collection_with_refs(chunk_size, min_chunk_size=None):
    mappings = utils.read_mappings()
    bernard = chunk_collection(
        load_bernard_documents(), chunk_size, min_chunk_size=min_chunk_size)
    NT = chunk_collection(
        load_NT_documents(), chunk_size, min_chunk_size=min_chunk_size)

    def find_NT_doc(NT_doc, ref):
        for (book, chapter, verse_num), *_ in NT_doc:
            if book == ref['book'] and \
               chapter == ref['chapter'] and \
               verse_num == ref['verse_num']:
                return True
        return False

    def find_bernard_doc(b_doc, span):
        b_ids, _, lemmas = zip(*b_doc)
        for b_id in b_ids:
            if b_id in span:
                return True
        return False

    # [ (ref_type, ref, span), bernard_doc_id, NT_doc_id ]
    refs = []
    for ref_id, (ref_type, ref, span) in tqdm.tqdm(enumerate(utils.read_refs())):
        try:
            parsed_ref = utils.build_bible_ref(ref, mappings)

            # search NT
            for NT_doc_id, NT_doc in enumerate(NT):
                if find_NT_doc(NT_doc, parsed_ref):
                    break
            else:
                print("Couldn't find NT doc for ref", parsed_ref)

            # search bernard
            for b_doc_id, b_doc in enumerate(bernard):
                if find_bernard_doc(b_doc, span):
                    break
            else:
                print("Couldn't find bernard doc for ref", parsed_ref)

            refs.append((ref_id, b_doc_id, NT_doc_id))

        except KeyError:
            print("Missing key for ref", ref)

    # fix bernard docs
    b_doc_ids = {}
    for b_doc_id in range(len(bernard)):
        b_ids, _, lemmas = zip(*bernard[b_doc_id])
        bernard[b_doc_id] = ' '.join(lemmas)
        # assume all bernard_ids are from the same sermo
        #   'lat.w.Sermo62.1.18',
        b_doc_ids[b_doc_id] = b_ids[0].split('.')[2]

    # fix NT docs
    NT_doc_ids = {}
    for NT_doc_id in range(len(NT)):
        book, _, lemmas = zip(*NT[NT_doc_id])
        NT[NT_doc_id] = ' '.join(lemmas)
        book, _, _ = zip(*book)
        book, *_ = book
        NT_doc_ids[NT_doc_id] = book

    return (bernard, NT), refs, (b_doc_ids, NT_doc_ids)


def read_docs(path, stopwords=None):
    with open(path) as f:
        for line in f:
            doc_id, doc = line.strip().split('\t')
            if stopwords is not None:
                doc = ' '.join([w for w in doc.split() if w not in stopwords])
            yield doc_id, doc


def read_refs(path):
    with open(path) as f:
        for line in f:
            ref_id, b_doc, NT_doc = line.strip().split('\t')
            yield int(ref_id), int(b_doc), int(NT_doc)


def fit_model(docs,
              # model parameters
              k=10,
              # vectorizer parameters
              min_df=1, max_df=1.0, max_features=None, **kwargs):

    counter = TfidfVectorizer(min_df=2, max_features=max_features)
    M = counter.fit_transform(docs)
    model = NMF(n_components=k, init='nndsvd', **kwargs)
    W = model.fit_transform(M)
    H = model.components_

    return model, counter, M, W, H


def get_descriptors(H, terms, topic_index, top_terms=10):
    descriptor_indices = H[topic_index, :].argsort()[::-1]
    descriptors = []
    for t in descriptor_indices[:top_terms]:
        descriptors.append(terms[t])
    return descriptors


def plot_descriptors(H, terms, topic_index, top_terms=10):
    descriptor_indices = H[topic_index, :].argsort()[::-1]
    descriptors = []
    descriptor_weights = []
    for term in descriptor_indices[:top_terms]:
        descriptors.append(terms[term])
        descriptor_weights.append(H[topic_index, term])

    descriptors.reverse()
    descriptor_weights.reverse()

    plt.figure(figsize=(12, 6))
    plt.barh(np.arange(len(descriptor_weights)),
             descriptor_weights,
             align='center',
             tick_label=descriptors)
    plt.tick_params(labelsize=14)
    plt.show()


def get_topic_coherence(descriptor_terms, similarity_fn):
    coherence = 0.0
    n = 0
    for i in range(len(descriptor_terms)):
        term_a = descriptor_terms[i]
        for j in range(i + 1, len(descriptor_terms)):
            term_b = descriptor_terms[j]
            try:
                coherence += similarity_fn(term_a, term_b)
                n += 1
            except Exception:
                pass
    return coherence / n


def get_model_coherence(similarity_fn, H, terms, top_terms=10):
    k = H.shape[0]
    coherence = 0
    for topic in range(k):
        descriptor_terms = get_descriptors(H, terms, topic, top_terms=top_terms)
        coherence += get_topic_coherence(descriptor_terms, similarity_fn)
    return coherence / k


class WordSimilarity:
    def __init__(self, path):
        words = {}
        with open(path) as f:
            for line in f:
                word, *vec = line.strip().split()
                words[word] = np.array([float(v) for v in vec])

        self.words = words

    def get_similarity(self, term_a, term_b):
        return 1-distance.cosine(self.words[term_a], self.words[term_b])


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', default=2300, type=int)
    parser.add_argument('--min_chunk_size', default=200, type=int)
    args = parser.parse_args()

    (bernard, NT), refs, (b_doc_ids, NT_doc_ids) = get_collection_with_refs(
        args.chunk_size, min_chunk_size=args.min_chunk_size)

    outfolder = 'output/topic-docs'

    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    with open(os.path.join(outfolder, 'bernard.csv'), 'w') as f:
        for doc_id, doc in enumerate(bernard):
            f.write(b_doc_ids[doc_id] + '\t' + doc + '\n')

    with open(os.path.join(outfolder, 'NT.csv'), 'w') as f:
        for doc_id, doc in enumerate(NT):
            f.write(NT_doc_ids[doc_id] + '\t' + doc + '\n')

    with open(os.path.join(outfolder, 'refs.csv'), 'w') as f:
        for ref_id, b_doc, NT_doc in refs:
            f.write('\t'.join([str(ref_id), str(b_doc), str(NT_doc)]) + '\n')
