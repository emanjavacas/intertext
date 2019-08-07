
import sys
import fileinput

with fileinput.FileInput() as inp:
    for line in inp:
        line = line.strip().replace('\xad', '')
        line = line.replace('æ', 'ae') \
                   .replace('œ', 'oe') \
                   .replace('Æ', 'Ae') \
                   .replace('Œ', 'Oe') \
                   .replace('ӕ', 'ae')
        print(line, file=sys.stdout)
