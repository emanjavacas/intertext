
import re
import glob

from intertext.patrologia import utils


if __name__ == '__main__':
    NT = {}
    for book, chapter, verse, token, lemma in utils.read_NT_lines():
        NT[book, chapter, verse] = token

    context = 100

    for f in glob.glob('patrologia/output/refs/*/*'):
        with open(f) as f:
            text = f.read()
            for m in re.finditer(utils.RE_REF, text):
                (start, end), ref = m.span(), m.group()
                left = text[start-context: start]
                right = NT[utils.decode_ref(ref)]
                print('\t'.join([left, ref, right]) + '\n')
