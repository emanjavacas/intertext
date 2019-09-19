
import os
import collections
import glob
import string


PATH = 'patrologia/output/raw/'

def strip_punctuation(w):
    return w.translate(str.maketrans('', '', string.punctuation))


def read_file(f, skip_hyphens=False):
    with open(f) as f:
        skip = 0
        for line in f:
            line = line.strip().split()
            for w1, w2 in zip(line, line[1:]):
                if skip:
                    skip -= 1
                    continue
                if skip_hyphens and w1.endswith('-'):
                    skip = 1
                    continue
                yield w1
            if not skip:
                yield w2


def get_vocab():
    vocab = collections.Counter()
    for f in glob.glob(PATH + '*/*.xml'):
        for w in read_file(f, skip_hyphens=True):
            vocab[strip_punctuation(w).lower()] += 1
    return vocab


def split_pre(w):
    prefix = ''
    w = list(w)
    while w and not w[0].isalpha():
        prefix += w.pop(0)
    return prefix, ''.join(w)


def split_post(w):
    post = ''
    w = list(w)
    while w and not w[-1].isalpha():
        post = w.pop(-1) + post
    return post, ''.join(w)


def merge_file(f, vocab):
    text = list(read_file(f))
    processed = []
    skip = 0
    for w1, w2 in zip(text, text[1:]):
        if skip:
            skip = False
            continue

        if w1.endswith('-'):
            merge = strip_punctuation(w1[:-1] + w2)
            if vocab.get(merge.lower(), 0) > 0:
                (prefix, w1), (suffix, w2) = split_pre(w1), split_post(w2)
                processed.append(prefix + merge + suffix)
                skip = True
            else:
                processed.append(w1)
        else:
            processed.append(w1)

    if not skip:
        processed.append(w2)

    return processed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='patrologia/output/merged/')
    args = parser.parse_args()

    vocab = get_vocab()

    for f in glob.glob(PATH + '*/*.xml'):

        parent = os.path.basename(os.path.dirname(f))
        parent = os.path.join(args.target, parent)
        if not os.path.isdir(parent):
            os.makedirs(parent)

        with open(os.path.join(parent, os.path.basename(f)), 'w') as outf:
            processed = merge_file(f, vocab)
            outf.write(' '.join(processed))
