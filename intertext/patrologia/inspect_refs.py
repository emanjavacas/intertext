
import re
import glob

import intertext.patrologia.utils as p_utils
from intertext import utils


if __name__ == '__main__':
    NT = {}
    for book, chapter, verse, token, lemma in utils.read_NT_lines():
        NT[book, chapter, verse] = token

    context = 100

    for f in glob.glob('output/patrologia/refs/*/*'):
        with open(f) as f:
            text = f.read()
            for m in re.finditer(p_utils.RE_REF, text):
                (start, end), ref = m.span(), m.group()
                left = text[start-context: start]
                right = NT[p_utils.decode_ref(ref)]
                print('\t'.join([left, ref, right]) + '\n')
