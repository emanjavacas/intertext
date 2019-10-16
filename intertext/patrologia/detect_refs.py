
import os
import glob
import collections
import re
import numpy as np

import Levenshtein

from intertext import utils
from intertext.patrologia.utils import encode_ref


RE_REF = r"([ivcxl]+ )?([a-z]+) ?[\.,·]? ?[\.,·]? ([icvxl]+) ?[\.,·]? ?([0-9]+|[icvxl]+)"


def get_refs(path='patrologia/refs.txt'):
    with open(path) as f:
        for line in f:
            filename, *ref = line.strip().split(':')
            ref = ':'.join(ref)
            yield filename, ref


def extract_ref(ref):
    # remove parenthses, lower and remove trailing space
    ref = ref.replace("(", "").replace(")", "").lower().strip()
    # normalize whitespace
    ref = ' '.join(ref.split())

    if ";" in ref:
        refs = ref.split(";")
        output = []
        for ref in refs:
            ref = ref.strip()
            ref = extract_ref(ref)
            if ref is not None:
                output.append(ref)
        return output

    m = re.match(RE_REF, ref)
    if m is not None:
        return tuple(g.strip() if g is not None else g for g in m.groups())


def get_levenshtein_mapping(counter):
    strings, _ = zip(*counter.most_common())
    dists = np.zeros((len(strings), len(strings)))
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            dists[i, j] = Levenshtein.distance(strings[i], strings[j])

    mapping = {}
    max_dist = max(len(s) for s in strings)
    dists[np.where(dists == 0)] = max_dist
    for i, j in enumerate(dists.argmin(axis=1)):
        if counter[strings[i]] >= counter[strings[j]]:
            a, b = strings[j], strings[i]
        else:
            a, b = strings[i], strings[j]
        mapping[a] = mapping.get(b, b)
    return mapping


def extract_refs(path):
    names, refs = zip(*get_refs(path=path))
    p_refs = []
    for r in refs:
        p_r = extract_ref(r)
        if p_r is None:
            continue
        if isinstance(p_r, tuple):
            p_refs.append(p_r)
        elif isinstance(p_r, list):
            for p_r in p_r:
                p_refs.append(p_r)

    return p_refs


def merge_refs(refs):
    counter = collections.defaultdict(collections.Counter)
    for book_num, book, chapter, verse in refs:
        counter["book_num"][book_num] += 1
        counter["book"][book] += 1
        counter["chapter"][chapter] += 1
        counter["verse"][verse] += 1

    return counter


def read_mapping(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            if line.startswith('?') or line.startswith('#'):
                continue
            a, b = line.strip().split()
            if a in mapping:
                raise ValueError("Duplicate entry in mapping file", a)
            mapping[a] = b
    return mapping


def apply_mapping(counts, mapping):
    output = collections.Counter()
    for k, c in counts.items():
        output[mapping.get(k, k)] += c
    return output


def roman_to_int(roman):
    values = {
        'M': 1000,
        'D': 500,
        'C': 100,
        'L': 50,
        'X': 10,
        'V': 5,
        'I': 1}

    roman = roman.upper()

    numbers = []
    for char in roman:
        numbers.append(values[char])

    if len(roman) == 1:
        return values[roman]

    total = 0
    for num1, num2 in zip(numbers, numbers[1:]):
        if num1 >= num2:
            total += num1
        else:
            total -= num1

    return total + num2


class BibleRef:
    def __init__(self):
        self.fixes = read_mapping('intertext/patrologia/book.mapping')
        self.mapping = utils.read_mappings('intertext/patrologia/bible.mapping')

    def map(self, ref):
        book_num, book, chapter, verse = ref
        book = self.mapping[self.fixes.get(book, book)]
        if book_num is not None:
            book = str(roman_to_int(book_num)) + ' ' + book
        chapter = str(roman_to_int(chapter))
        if not verse.isdigit():
            verse = str(roman_to_int(verse))
        return book, chapter, verse

    def find_refs(self, text):
        for m in re.finditer("\\([^)]+\\)", text):
            start, end = m.span()
            ref = m.group()
            ref = extract_ref(ref)
            # fail to find match
            if not ref:
                continue
            # single ref
            if isinstance(ref, tuple):
                try:
                    book, chapter, verse = self.map(ref)
                    yield (book, chapter, verse), (start, end)
                except Exception:
                    continue
            # multi ref
            else:
                refs = []
                for ref in ref:
                    try:
                        book, chapter, verse = self.map(ref)
                        refs.append((book, chapter, verse))
                    except Exception:
                        continue
                if refs:
                    yield refs, (start, end)


def encode_refs(ref):
    return ' '.join(map(encode_ref, ref))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='output/patrologia/refs')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    NT = {}
    for book, chapter, verse, token, lemma in utils.read_NT_lines():
        NT[book, chapter, verse] = token

    bible_ref = BibleRef()
    found = 0
    not_in_nt = 0

    for f in glob.glob('output/patrologia/merged/*/*'):

        parent = os.path.basename(os.path.dirname(f))
        parent = os.path.join(args.target, parent)
        if not os.path.isdir(parent):
            os.makedirs(parent)
        path = os.path.join(parent, os.path.basename(f))

        with open(f) as inf, open(path, 'w') as outf:
            text = ' '.join(inf.read().split())
            refs = list(bible_ref.find_refs(text))[::-1]
            for ref, (start, end) in refs:
                # multi-ref
                if isinstance(ref, list):
                    # filter out those not in NT
                    refs = []
                    for ref in ref:
                        if ref in NT:
                            found += 1
                            refs.append(ref)
                        else:
                            not_in_nt += 1
                            if args.verbose:
                                print("missing ref in NT", ref)
                    if not refs:
                        continue
                    else:
                        ref = encode_refs(refs)
                # single ref
                else:
                    if ref not in NT:
                        not_in_nt += 1
                        if args.verbose:
                            print("missing ref in NT", ref)
                        continue
                    else:
                        found += 1
                        ref = encode_ref(ref)

                text = text[:start] + ' ' + ref + ' ' + text[end:]

            outf.write(' '.join(text.split()))

    print("Found {} refs. {} missing from NT".format(found, not_in_nt))
