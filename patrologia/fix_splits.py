
import os
import glob


with open('patrologia/vocab.csv') as f:
    vocab = {}
    for line in f:
        line = line.strip()
        c, w = line.split()
        vocab[w] = int(c)


def check(w1, w2):
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

    (prefix, w1), (post, w2) = split_pre(w1), split_post(w2)

    # skip only punctuation
    if not w1 or not w2:
        return

    if w1.endswith('-') and w1[:-1].lower() + w2.lower() in vocab:
        return [prefix + w1[:-1] + w2 + post]

    if (w1 + w2).lower() in vocab:
        if w1.lower() not in vocab:
            return [prefix + w1 + w2 + post]
        elif w2.lower() not in vocab:
            return [prefix + w1 + w2 + post]
        else:
            if vocab[(w1 + w2).lower()] > vocab[w1.lower()] \
               and vocab[(w1 + w2).lower()] > vocab[w2.lower()]:
                return [prefix + w1 + w2 + post]


target = 'patrologia/processed-hyphens'

for f in glob.glob('patrologia/processed/*/*'):
    *_, folder, filename = f.split('/')
    outputfolder = os.path.join(target, folder)
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)

    with open(f) as inf, open(os.path.join(outputfolder, filename), 'w+') as outf:
        for line in inf:

            line = line.strip().split()
            if len(line) < 2:
                outf.write(' '.join(line) + '\n')
                continue

            newline, idx = [], 0
            while idx < len(line):
                if idx == len(line) - 1:  # last word
                    newline.append(line[-1])
                    break

                w1, w2 = line[idx], line[idx + 1]
                words = check(w1, w2)
                if words is not None:
                    newline.extend(words)
                    idx += 2
                else:
                    newline.append(w1)
                    idx += 1
            newline = ' '.join(newline)
            outf.write(newline + '\n')
