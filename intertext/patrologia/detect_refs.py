
import os
import glob
import collections
import re
import numpy as np

import Levenshtein

from intertext import utils
import intertext.patrologia.utils as p_utils


RE_REF = (
    r"([ivcxl]+ )?"         # book num
    r"([a-z]+)"              # book
    r" ?[\.,·]? ?[\.,·]? "
    r"([icvxl]+)"            # chapter
    r" ?[\.,·]? "
    r"(i?[0-9]+|[icvxl]+)"    # verse
)

# TODO: extract multiple refs
# - (Coloss. III, 1 , 2) Colossians_3_1
# - (Isai. XL, 6-8) Isaiah_40_6
# - (Matth . vi, 20 et 21) Matthew_6_20
# - ( Act, VII, 58. et 59) Acts_7_58
# - ( Psal. XXXVI, 10, 55, 36) Psalms_36_10
RE_REF_1 = r"et ([0-9]+|[icvxl]+)"
RE_REF_2 = r"- ?([1-9][0-9]*)"
RE_REF_3 = r"([\.,] ?([1-9][0-9]*))+"
RE_REF_COMPLEX = r"(?P<rest> ?[\.,]?{}".format(
    '(' + '|'.join([RE_REF_1, RE_REF_2, RE_REF_3]) + '))?')
RE_REF += RE_REF_COMPLEX

# TODO: split on other than just ";":
# - (Matth. XXII, 23 32, et Luc. xx, 27-58) Matthew_22_23
# - ( Ephes. IV, 15, et Coloss. I, 18 ) Ephesians_4_15

def get_refs(path='patrologia/refs.txt'):
    with open(path) as f:
        for line in f:
            filename, *ref = line.strip().split(':')
            ref = ':'.join(ref)
            yield filename, ref


def extract_ref(ref):
    # remove parentheses, lower and remove trailing space
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
        # base groups
        book_num, book, chapter, verse, *_ = [
            (g or '').strip() or None for g in m.groups()]

        # check rest
        rest = m.groupdict()['rest']

        if rest is not None:
            rest = rest.strip()
            tup = (book_num, book, chapter)

            # et
            if re.match(RE_REF_1, rest):
                new_verse = re.search(r"([0-9]+|[icvxl]+)", rest).group()
                return [tup + (verse,), tup + (new_verse,)]

            # hyphen range
            elif re.match(RE_REF_2, rest):
                if not verse.isdigit():
                    # assume mistake
                    return
                start, stop = int(verse), int(re.search(r"([0-9]+)", rest).group())
                # ignore spans larger than 15
                if start > stop or start == stop or stop - start > 15:
                    return
                output = []
                for verse in range(start, stop + 1):
                    output.append(tup + (str(verse), ))
                return output

            # comma separated
            elif re.match(RE_REF_3, rest):
                tup = (book_num, book, chapter)
                output = []
                output.append(tup + (verse,))
                for verse in re.finditer(r"([0-9]+|[icvxl]+)", rest):
                    output.append(tup + (verse.group(),))
                return output
            else:
                print('Warning, unmatched complex regex: "{}"'.format(rest))

        return book_num, book, chapter, verse


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
        RE_REF_DETECT = r"( [i]+ )?({})[\.,·]? [\.,·]?[ ]?" + \
                        r"([icvxl]+)[ ]?[\.,·][ ]?([0-9]+|[icvxl]+)"
        self.RE_REF_DETECT = RE_REF_DETECT.format(
            '|'.join(list(self.fixes) + list(self.mapping)))
        self.has_book_num = set(
            ['Kings', 'John', 'Samuel', 'Timothy', 'Peter',
             'Maccabees', 'Thessalonians', 'Chronicles', 'Corinthians'])

    def map(self, ref):
        book_num, book, chapter, verse, *_ = ref
        book = self.mapping[self.fixes.get(book, book)]
        if book_num is not None:
            book = str(roman_to_int(book_num)) + ' ' + book
        chapter = str(roman_to_int(chapter))
        if verse.startswith('i') and verse[1:].isdigit():
            verse = '1' + verse[1:]
        elif not verse.isdigit():
            verse = str(roman_to_int(verse))
        return book, chapter, verse

    def find_refs(self, text):
        for m in re.finditer("\\([^)]{,100}\\)", text):
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

    def detect_refs(self, text):
        text = ' '.join(text.lower().strip().split())
        for ref in re.finditer(self.RE_REF_DETECT, text):
            book_num, book, chapter, verse = ref.groups()
            if book_num is not None:
                book_num = book_num.strip()
                try:
                    if self.mapping[self.fixes.get(book, book)] not in self.has_book_num:
                        book_num = None
                except KeyError:
                    continue
            try:
                book, chapter, verse = self.map((book_num, book, chapter, verse))
                start, end = ref.span()
                yield (book, chapter, verse), (start, end)
            except Exception:
                continue


def encode_refs(ref):
    return ' '.join(map(p_utils.encode_ref, ref))


def read_refs():
    refs = []
    for f in glob.glob('output/patrologia/refs/*/*'):
        with open(f) as inp:
            text = inp.read()
            for ref in re.finditer(p_utils.RE_REF, text):
                refs.append(p_utils.decode_ref(ref.group()))
    return refs


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
    found = found_detect = 0
    not_in_nt = not_in_nt_detect = 0

    for f in glob.glob('output/patrologia/merged/*/*'):

        parent = os.path.basename(os.path.dirname(f))
        parent = os.path.join(args.target, parent)
        if not os.path.isdir(parent):
            os.makedirs(parent)
        path = os.path.join(parent, os.path.basename(f))

        with open(f) as inf, open(path, 'w') as outf:
            text = ' '.join(inf.read().split())
            # 1. Find references around parentheses
            refs = list(bible_ref.find_refs(text))[::-1]
            for ref, (start, end) in refs:
                # 1.1. multi-ref
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
                # 1.2. single ref
                else:
                    if ref not in NT:
                        not_in_nt += 1
                        if args.verbose:
                            print("missing ref in NT", ref)
                        continue
                    else:
                        found += 1
                        ref = p_utils.encode_ref(ref)

                text = text[:start] + ' ' + ref + ' ' + text[end:]

            # # Warning: this is quite noisy, stays disabled by default
            # # 2. Find references on free running text under stricter conditions
            # refs = list(bible_ref.detect_refs(text))[::-1]
            # for ref, (start, end) in refs:
            #     if ref not in NT:
            #         not_in_nt_detect += 1
            #         if args.verbose:
            #             print("missing ref in NT", ref)
            #         continue
            #     else:
            #         found_detect += 1
            #         ref = p_utils.encode_ref(ref) + '___'
            #         text = text[:start] + ' ' + ref + ' ' + text[end:]

            outf.write(' '.join(text.split()))

    print("Found {} refs. {} missing from NT".format(found, not_in_nt))
    print("Detected {} refs. {} missing from NT".format(found_detect, not_in_nt_detect))


# NT = {}
# for book, chapter, verse, token, lemma in utils.read_NT_lines():
#     NT[book, chapter, verse] = token
# bible_ref = BibleRef()
# refs = []
# for idx, f in enumerate(glob.glob('output/patrologia/merged/*/*')):
#     if idx == 2:
#         break
#     with open(f) as inp:
#         src_text = ' '.join(inp.read().split())
#         text = src_text
#         refs = list(bible_ref.find_refs(src_text))[::-1]
#         print(len(refs))
#         for ref, (start, end) in refs:
#             # 1.1. multi-ref
#             if isinstance(ref, list):
#                 # filter out those not in NT
#                 refs = []
#                 for ref in ref:
#                     if ref in NT:
#                         refs.append(ref)
#                 if not refs:
#                     continue
#                 else:
#                     ref = encode_refs(refs)
#             # 1.2. single ref
#             else:
#                 if ref not in NT:
#                     continue
#                 else:
#                     ref = p_utils.encode_ref(ref)

#             print(text[start:end], ref)
#             text = text[:start] + ' ' + ref + ' ' + text[end:]

# def inline_diff(a, b):
#     import difflib
#     matcher = difflib.SequenceMatcher(None, a, b)
#     def process_tag(tag, i1, i2, j1, j2):
#         if tag == 'replace':
#             return '{' + matcher.a[i1:i2] + ' -> ' + matcher.b[j1:j2] + '}'
#         if tag == 'delete':
#             return '{- ' + matcher.a[i1:i2] + '}'
#         if tag == 'equal':
#             return matcher.a[i1:i2]
#         if tag == 'insert':
#             return '{+ ' + matcher.b[j1:j2] + '}'
#         assert false, "Unknown tag %r"%tag
#     return ''.join(process_tag(*t) for t in matcher.get_opcodes())

# # inline_diff(src_text, text)
