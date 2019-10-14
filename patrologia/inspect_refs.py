
import re
import glob
import utils


def decode_ref(ref):
    book, chapter, verse = ref.split('_')
    book = ' '.join(book.split('-'))
    return book, chapter, verse


REF_RE = r'([0-9]+-)?([A-Z]*[a-z]+-)*[A-Z][a-z]+_[0-9]+_[0-9]+'


if __name__ == '__main__':
    NT = {}
    for book, chapter, verse, token, lemma in utils.read_NT_lines():
        NT[book, chapter, verse] = token

    context = 100

    for f in glob.glob('patrologia/output/refs/*/*'):
        with open(f) as f:
            text = f.read()
            for m in re.finditer(REF_RE, text):
                (start, end), ref = m.span(), m.group()
                left = text[start-context: start]
                right = NT[decode_ref(ref)]
                print('\t'.join([left, ref, right]) + '\n')
