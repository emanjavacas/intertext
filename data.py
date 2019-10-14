
import collections
import itertools

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
