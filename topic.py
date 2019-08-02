
import collections
import itertools

import tqdm

import utils


def parse_id(string):
    "'lat.w.Sermo62.1.16'"
    _, wordtype, sermo, chapter, verse = string.split('.')

    return {'wordtype': wordtype,
            'sermo': sermo,
            'chapter': chapter,
            'verse': verse}


def load_bernard_documents():
    sermos = []
    for sermo, group in itertools.groupby(filter(None, utils.read_bernard_lines()),
                                          key=lambda tup: parse_id(tup[0])['sermo']):
        sermo_data = []
        for idx, word, _, _, lemma in group:
            if '.pc.' in idx:
                continue
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


def get_chunks(words, max_chunk_size, min_chunk_size=None):
    for i in range(0, len(words), max_chunk_size):
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
        

if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', default=2300, type=int)
    parser.add_argument('--min_chunk_size', default=200, type=int)
    args = parser.parse_args()

    (bernard, NT), refs, (b_doc_ids, NT_doc_ids) = get_collection_with_refs(
        args.chunk_size, min_chunk_size=args.min_chunk_size)

    outfolder = 'topic-docs'

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
            f.write('\t'.join([ref_id, b_doc, NT_doc]) + '\n')
