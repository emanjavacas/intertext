
import collections
import itertools
import os
import numpy as np


def lemmatize_pie(model, sent, use_beam=True, beam_width=12, device='cpu'):
    import pie
    inp, _ = pie.data.pack_batch(model.label_encoder, [sent], device=device)
    return model.predict(inp, "lemma", use_beam=use_beam, beam_width=beam_width)


def lemmatize_treetagger(model, sent):
    token, pos, lemma = zip(*[line.split('\t') for line in model.tag_text(sent)])
    return {'token': token, 'pos': pos, 'lemma': lemma}


def load_embeddings(words, path):
    print("loading {} words from {}".format(len(words), path))
    embs, vocab = [], []
    with open(path) as f:
        next(f)
        for line in f:
            word, *vec = line.split()
            if word in words:
                embs.append(list(map(float, vec)))
                vocab.append(word)

    print("Found {}/{} words".format(len(vocab), len(words)))
    return np.array(embs), vocab


def parse_ref(ref):
    _, book, chapter, verse = ref.split('_')
    *book_num, book = book.split('%20')
    book_num = book_num[0] if book_num else None
    return book_num, book, chapter, verse


def parse_id(s):
    "'lat.w.Sermo62.1.16'"
    _, wordtype, sermo, chapter, verse = s.split('.')

    return {'wordtype': wordtype,
            'sermo': sermo,
            'chapter': chapter,
            'verse': verse}


def read_refs(path='output/bernard/refs.csv'):
    with open(path) as f:
        for line in f:
            line = line.strip()
            ref_type, ref, span = line.split('\t')
            span = set(idx for idx in span.split('-'))
            ref = parse_ref(ref)

            yield ref_type, ref, span


def read_mappings(path='output/bernard_bible_mappings.csv'):
    with open(path) as f:
        mappings = {}
        for line in f:
            a, b = line.strip().split('\t')
            mappings[a] = b

    return mappings


def read_bernard_lines(path='output/bernard/docs', drop_pc=False, lower=False):
    for f in os.listdir(path):
        with open(os.path.join(path, f)) as f:
            for line in f:
                line = line.strip()
                if line:
                    wid, token, tt_lemma, pos, pie_lemma = line.split('\t')
                    if drop_pc and '.pc.' in wid:
                        continue
                    if lower:
                        token = token.lower()
                    yield wid, token, tt_lemma, pos, pie_lemma
                else:
                    yield None


def read_NT_lines(path='output/NT.csv', lower=False):
    with open(path) as f:
        for line in f:
            # book, chapter, verse_num, verse(, lemma)
            book, chapter, verse_num, verse, lemma = line.strip().split('\t')
            if lower:
                verse = verse.lower()

            yield book, chapter, verse_num, verse, lemma


def read_stopwords(path='all.stop'):
    with open(path) as f:
        return set(w.strip() for w in f)


def collect_refs():
    tokens, lemmas, poses = [], [], []
    id2idx = {}
    for line in read_bernard_lines():
        if line is None:
            continue

        bibl_id, tok, lem, pos = line
        assert bibl_id not in id2idx, "got known id {}".format(bibl_id)
        id2idx[bibl_id] = len(tokens)
        tokens.append(tok)
        lemmas.append(lem)
        poses.append(pos)

    missing = 0
    for ref_type, ref, span in read_refs():
        try:
            idxs = sorted(id2idx[idx] for idx in span)
            text = ' '.join([tokens[idx] for idx in idxs])

            yield ref_type, ref, span, text

        except KeyError:
            missing += 1

    print("Missing {} references".format(missing))


def build_bible_ref(ref, mappings):
    book_num, book, chapter, verse_num = ref
    book = mappings[book]
    if book_num is not None:
        book = book_num + ' ' + book
    return {'book': book, 'chapter': chapter, 'verse_num': verse_num}


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
            filter(None, read_bernard_lines(drop_pc=True, lower=True)),
            key=lambda tup: parse_id(tup[0])['sermo']):

        group = list(group)

        for idx in range(0, len(group), nwords - overlap):
            # stop when end is reached
            if idx + nwords == len(group):
                break

            ids, tokens, lemmas, _, _ = zip(*group[idx: idx + nwords])
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
    lines = sorted(read_NT_lines(), key=lambda tup: tup[0])

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
    mappings = read_mappings()
    for ref_type, (book_num, book, chapter, verse_num), span in read_refs():
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


# mappings = read_mappings()
# NT = {}
# for book, chapter, verse_num, verse, *_ in read_NT_lines():
#     NT[book, chapter, verse_num] = verse
# refs = list(collect_refs())
# missing = 0
# output = []
# for ref in refs:
#     ref_type, (book_num, book, chapter, verse_num), span, text = ref
#     try:
#         NT_book = mappings[book]
#         if book_num is not None:
#             NT_book = book_num + ' ' + NT_book
#         target = NT[NT_book, chapter, verse_num]
#         output.append((ref, target))
#     except Exception:
#         missing += 1

# by_type = collections.defaultdict(list)
# for ref, target in output:
#     (ref_type, (book_num, book, chapter, verse_num), text) = ref
#     by_type[ref_type].append((ref, target))
