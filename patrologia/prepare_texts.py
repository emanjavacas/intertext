
import os
import string
import collections
from lxml import etree
import glob

PATH = 'patrologia/corrected/'
PARSER = etree.XMLParser(recover=True)
NSMAP = {'tei': 'http://www.tei-c.org/ns/1.0'}


def parse_tree(fpath):
    tree = etree.parse(
        # inp.read().encode('utf-8'),
        fpath,
        parser=PARSER)
    return tree


def is_linebreak(child):
    return child.tag == '{{{}}}lb'.format(NSMAP['tei'])


def process_text(t):
    # normalize whitespace
    t = ' '.join(t.strip().split())

    # replace characters
    t = t.replace('\xad', '') \
         .replace('æ', 'ae') \
         .replace('œ', 'oe') \
         .replace('Æ', 'Ae') \
         .replace('Œ', 'Oe') \
         .replace('ӕ', 'ae')

    return t


def extract_text(text, return_lbs=False):
    plain = []

    for d in text.iterdescendants():
        if return_lbs:
            if is_linebreak(d) and d.tail:
                t = process_text(d.tail)
                if t:
                    # maybe merge
                    plain.extend(['<lb>'] + t.split())
                continue

        if d.text:
            t = process_text(d.text).split()
            if t:
                plain.extend(t)
        if d.tail:
            t = process_text(d.tail).split()
            if t:
                plain.extend(t)

    return plain


def extract_lbs(text):
    lbs = []
    plain = extract_text(text, return_lbs=True)
    for w1, lb, w2 in zip(plain, plain[1:], plain[2:]):
        if lb == '<lb>':
            lbs.append((w1, w2))

    return lbs


def strip_punctuation(w):
    return w.translate(str.maketrans('', '', string.punctuation))


def get_vocab():
    counter = collections.Counter()
    for f in glob.glob(PATH + '*/*.xml'):
        tree = parse_tree(f)
        text = tree.xpath("//tei:text", namespaces=NSMAP)[0]
        plain = extract_text(text, return_lbs=True)
        for w1, lb, w2 in zip(plain, plain[1:], plain[2:]):
            if lb != '<lb>':
                w = strip_punctuation(w1)
                if w:
                    counter[w.lower()] += 1
    return counter


def get_merges(vocab):
    lbs = []
    for f in glob.glob(PATH + '*/*.xml'):
        tree = parse_tree(f)
        lbs.extend(extract_lbs(tree.xpath("//tei:text", namespaces=NSMAP)[0]))

    merges = []
    for w1, w2 in lbs:
        w1_, w2_ = strip_punctuation(w1).lower(), strip_punctuation(w2).lower()
        if w1_ and w2_:
            merge = (w1_ + w2_)
            merges.append((
                merge, vocab.get(merge, 0),
                w1_, vocab.get(w1_, 0),
                w2_, vocab.get(w2_, 0)))

    merges = list(set(merges))
    merges = sorted(merges, key=lambda tup: tup[1], reverse=True)
    by_freq = collections.defaultdict(list)
    for tup in merges:
        by_freq[tup[1]].append(tup)

    binned = collections.defaultdict(list)
    maxlen = 10000
    current, counts = [], 0
    for i in sorted(by_freq, reverse=True):
        if counts >= maxlen:
            if len(current) >= 2:
                start, *_, stop = current
                binned[start, stop] = [i for b in current for i in by_freq[b]]
            else:
                binned[current[0]] = by_freq[current[0]]
            current, counts = [], 0
        current.append(i)
        counts += len(by_freq[i])

    by_bin = {}
    for bins, merges in binned.items():
        for merge, *_ in merges:
            by_bin[merge] = bins

    return binned, by_bin


class Merger:
    def __init__(self, vocab, reserved=('et', 'ut', 'sive', 'est')):
        self.vocab = vocab
        _, by_bin = get_merges(vocab)
        self.by_bin = by_bin
        self.reserved = reserved

    @staticmethod
    def split_pre(w):
        prefix = ''
        w = list(w)
        while w and not w[0].isalpha():
            prefix += w.pop(0)
        return prefix, ''.join(w)

    @staticmethod
    def split_post(w):
        post = ''
        w = list(w)
        while w and not w[-1].isalpha():
            post = w.pop(-1) + post
        return post, ''.join(w)

    @staticmethod
    def do_merge(w1, w2):
        (prefix, w1), (suffix, w2) = Merger.split_pre(w1), Merger.split_post(w2)
        w1, w2 = strip_punctuation(w1), strip_punctuation(w2)
        return prefix + w1 + w2 + suffix

    def merge(self, w1, w2):
        w1_strip = strip_punctuation(w1).lower()
        w2_strip = strip_punctuation(w2).lower()

        if not (w1_strip and w2_strip):
            return

        if w1_strip in self.reserved or w2_strip in self.reserved:
            return

        if w2_strip == 'que':
            return

        bins = self.by_bin.get(w1_strip + w2_strip, None)
        if not bins:            # unknown merge
            return

        if bins == 1:
            min_w_freq, max_w_freq = sorted([self.vocab[w1_strip], self.vocab[w2_strip]])
            if max_w_freq > 40000 and min_w_freq < 5:
                return Merger.do_merge(w1, w2)
            else:
                return

        elif (w1_strip + w2_strip in self.vocab):
            return Merger.do_merge(w1, w2)
        else:
            return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='patrologia/output/raw/')
    args = parser.parse_args()

    vocab = get_vocab()
    merger = Merger(vocab)
    for f in glob.glob(PATH + '*/*.xml'):
        tree = parse_tree(f)
        text = tree.xpath("//tei:text", namespaces=NSMAP)[0]
        text = extract_text(text, return_lbs=True)
        processed = []
        skip = 0
        for w1, lb, w2 in zip(text, text[1:], text[2:]):
            if skip:
                skip -= 1
                continue
            if lb == '<lb>':
                merge = merger.merge(w1, w2)
                if merge:
                    processed.append(merge)
                    skip = 2
                else:
                    processed.append(w1)
                    skip = 1
            else:
                processed.append(w1)

        processed.extend([lb, w2][:2-skip])

        parent = os.path.basename(os.path.dirname(f))
        parent = os.path.join(args.target, parent)
        if not os.path.isdir(parent):
            os.makedirs(parent)

        with open(os.path.join(parent, os.path.basename(f)), 'w') as f:
            f.write(' '.join(processed))


# def filter_freq(bins, min_freq=5, max_freq=40000):
#     valid, invalid = [], []
#     for merge in bins:
#         _, m_freq, w1, w1_freq, w2, w2_freq = merge
#         min_w_freq, max_w_freq = sorted([w1_freq, w2_freq])
#         if max_w_freq > max_freq and min_w_freq < min_freq:
#             valid.append(merge)
#         else:
#             invalid.append(merge)
#     return valid, invalid

# freqs_a, freqs_b = list(filter_freq(binned[1], max_freq=40000))

# fn = 10
# for f in glob.glob(PATH + '*/*.xml'):
#     tree = parse_tree(f)
#     if fn == 0:
#         break
#     fn -= 1g

# text = tree.xpath("//tei:text", namespaces=NSMAP)[0]
# plain = extract_text(text, return_lbs=True)
# lbs = []
# for w1, lb, w2 in zip(plain, plain[1:], plain[2:]):
#     if lb == '<lb>':
#         lbs.append((w1, w2))
