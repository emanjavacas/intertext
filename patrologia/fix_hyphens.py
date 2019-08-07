
import sys
import fileinput

with fileinput.FileInput() as inp:
    last = None
    for line in inp:
        line = line.strip()
        if last is not None:
            line = last + line
            last = None
        if line.endswith('-') or line.endswith('—'):
            *line, last = line.split()
            line = ' '.join(line)
            last = last[:-1]
        print(line, file=sys.stdout)
