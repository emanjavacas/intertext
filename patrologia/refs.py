
import collections
import re
import numpy as np

import Levenshtein

import utils


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
            ref = extract_ref(ref)
            if ref is not None:
                output.append(ref)
        return output

    m = re.match(r"([ivcxl]+ )?([a-z]+) ?\. ([icvxl]+), ([0-9]+)", ref)
    if m is not None:
        return m.groups()


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


def extract_refs():
    names, refs = zip(*get_refs())
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
        self.fixes = read_mapping('patrologia/book.mapping')
        self.mapping = utils.read_mappings('patrologia/patrologia_bible_mappings.csv')

    def map(self, ref):
        book_num, book, chapter, verse = ref
        book = self.mapping[self.fixes.get(book, book)]
        if book_num is not None:
            book = str(roman_to_int(book_num)) + book
        chapter = str(roman_to_int(chapter))
        return book, chapter, verse

    def find_refs(self, text):
        for m in re.finditer("\\([^)]+\\)", text):
            start, end = m.span()
            ref = m.group()
            try:
                ref = extract_ref(ref)
                book, chapter, verse = self.map(ref)
                yield (book, chapter, verse), (start, end)
            except Exception as e:
                pass


NT = {}
for book, chapter, verse, token, lemma in utils.read_NT_lines():
    NT[book, chapter, verse] = token

bible_ref = BibleRef()

# refs = extract_refs()
# processed = []
# for ref in refs:
#     try:
#         processed.append(bible_ref.map(ref))
#     except:
#         pass

# found = []
# for ref in processed:
#     if ref in NT:
#         found.append((ref, NT[ref]))


import glob

found = 0
for f in glob.glob('patrologia/processed-hyphens/*/*'):
    with open(f) as f:
        text = ' '.join(f.read().split())
        for ref, (start, end) in bible_ref.find_refs(text):
            if ref not in NT:
                print("missing ref in NT", ref)
                continue

            found += 1
            print("Id: {}".format(found),
                  "\n\nref: ", ref,
                  "\n\ntext: ", ' '.join(text[max(0, start-150): start].strip().split()),
                  "\n\nbible:", NT[ref],
                  "\n\n****\n\n\n")
