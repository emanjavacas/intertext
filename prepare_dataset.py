
import os
import collections

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.model_selection import train_test_split

import retrieval
import random

random_state = 1001
random.seed(random_state)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('outputfile')
    parser.add_argument('--nwords', type=int, default=15)
    parser.add_argument('--overlap', type=int, default=7)
    parser.add_argument('--top_k', type=int, default=25)
    parser.add_argument('--num_neg', type=int, default=4)
    parser.add_argument('--use_lemmas', action='store_true')
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--context_words', type=int, default=0)
    args = parser.parse_args()

    field = 'lemmas' if args.use_lemmas else 'tokens'
    bernard, b_id2doc = retrieval.load_bernard(
        args.nwords, args.overlap, context_words=args.context_words)
    bernard, b_context = zip(*bernard)
    bernard = list(bernard)
    nt, nt_id2doc = retrieval.load_NT()
    nt, nt_context = zip(*nt)
    nt = list(nt)
    refs = list(retrieval.load_refs(b_id2doc, nt_id2doc))

    # use lemmas for the similarity computation
    tfidf = TfidfVectorizer(norm=None).fit(getattr(doc, 'lemmas') for doc in bernard + nt)
    src_embs = tfidf.transform(getattr(doc, 'lemmas') for doc in bernard).toarray()
    trg_embs = tfidf.transform(getattr(doc, 'lemmas') for doc in nt).toarray()
    D = pairwise_kernels(src_embs, trg_embs, metric='cosine', n_jobs=-1)
    D = D.argsort()

    # store refs by src for negative sampling later
    by_src = collections.defaultdict(set)
    if args.context_words > 0:
        print("Using context document of size {}".format(args.context_words))
        bernard = list(b_context)

    rows = []
    for ref_type, src, trg in refs:
        for src_i in src:
            rows.append([
                ref_type,
                getattr(bernard[src_i], field),
                getattr(nt[trg], field),
                str(1)
            ])
            by_src[src_i].add(trg)

    # negative examples
    for _ in range(args.num_neg * len(rows)):
        src = random.randint(0, len(D) - 1)
        top_k = D[src, -args.top_k:].tolist()
        neg = top_k.pop(random.randint(0, len(top_k) - 1))
        while neg in by_src[src]:
            neg = top_k.pop(random.randint(0, len(top_k) - 1))
        rows.append([
            'negative',
            getattr(bernard[src], field),
            getattr(nt[neg], field),
            str(0)
        ])

    train, rest = train_test_split(
        rows, stratify=[row[0] for row in rows], train_size=args.train_size,
        random_state=random_state)
    test, dev = train_test_split(
        rest, stratify=[row[0] for row in rest], train_size=0.5,
        random_state=random_state)

    def write_file(rows, infix, header=('ref_type', 'src', 'trg', 'label')):
        path = os.path.join('output', args.outputfile + '.{}.csv'.format(infix))
        with open(path, 'w+') as f:
            f.write('\t'.join(header) + '\n')
            for row in rows:
                f.write('\t'.join(row) + '\n')

    write_file(train, 'train')
    write_file(test, 'test')
    write_file(dev, 'dev')
