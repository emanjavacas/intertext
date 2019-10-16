
import os

from cltk.tokenize.latin.sentence import SentenceTokenizer

from intertext import parse_bernard

sentok = SentenceTokenizer()


def get_tokens(tree):
    token, pos, lemma, idx = [], [], [], []

    for t in tree.xpath('//tei:w | //tei:pc', namespaces=parse_bernard.NSMAP):
        token.append(t.text)
        idx.append(t.attrib['{http://www.w3.org/XML/1998/namespace}id'])
        if t.tag[-1] == 'w':
            p, l = t.attrib['ana'], t.attrib['lemma']
        else:
            p, l = 'PC', t.text
        pos.append(p)
        lemma.append(l)

    return token, pos, lemma, idx


def get_title(tree):
    short, large = tree.xpath('//tei:title', namespaces=parse_bernard.NSMAP)
    return '-'.join([short.text.upper().strip(), large.text.lower().strip()])


def detokenize(tokens, pos):
    sent = ''
    for t, p in zip(tokens, pos):
        if p == 'PC':
            sent += t
        else:
            sent += ' ' + t
    return sent.strip()


def get_sent_boundaries(tokens, sents):
    boundaries = []
    rev_tokens = tokens[::-1]
    boundary = 0

    for sent in sents:
        while sent:
            boundary += 1
            tok = rev_tokens.pop()
            assert tok == sent[:len(tok)], (tok, sent[:len(tok)])
            sent = sent[len(tok):].strip()

        boundaries.append(boundary)

    return boundaries


def apply_sent_boundaries(sent, boundaries):
    output = []
    last = 0
    for boundary in boundaries:
        output.append(sent[last:boundary])
        last = boundary
    return output


def prepare_docs(source, target):
    for fname, tree in parse_bernard.parse_dir(source):
        token, pos, lemma, idx = get_tokens(tree)
        ntokens = len(token)
        sents = sentok.tokenize(detokenize(token, pos))
        boundaries = get_sent_boundaries(token, sents)
        token = apply_sent_boundaries(token, boundaries)
        assert sum(len(t) for t in token) == ntokens
        lemma = apply_sent_boundaries(lemma, boundaries)
        pos = apply_sent_boundaries(pos, boundaries)
        idx = apply_sent_boundaries(idx, boundaries)

        with open(os.path.join(target, fname[:-4] + '.csv'), 'w') as f:
            for t, l, p, i in zip(token, lemma, pos, idx):
                for t_t, l_t, p_t, i_t in zip(t, l, p, i):
                    f.write('\t'.join([i_t, t_t, l_t, p_t]) + '\n')
                f.write('\n')


def export_refs(source, target):
    errors = 0
    with open(target, 'w') as f:
        for _, tree in parse_bernard.parse_dir('source/SCT1-5/'):
            for seg, note in parse_bernard.get_seg_notes(tree):
                span = []
                for it in seg.xpath('tei:w | tei:pc', namespaces=parse_bernard.NSMAP):
                    span.append(it.attrib['{http://www.w3.org/XML/1998/namespace}id'])
                span = '-'.join(span)
                link = parse_bernard.get_link(note)
                link_type = parse_bernard.get_link_type(link)
                try:
                    url, _ = parse_bernard.get_ptr(note)
                    # chunk off the url to save space
                    # http://www.biblindex.info/fr/biblical/content/ref/
                    url = url.split('/')[-1]
                except Exception:
                    errors += 1
                    continue
                f.write("{}\t{}\t{}\n".format(link_type, url, span))

    print("Got ", errors, " parsing errors")


if __name__ == '__main__':
    source = 'source/SCT1-5/'
    target = 'output/bernard/docs'
    ref_target = 'output/bernard/refs.csv'
    if not os.path.isdir(target):
        os.makedirs(target)
    prepare_docs(source, target)
    export_refs(source, ref_target)
