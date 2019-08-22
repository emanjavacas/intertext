
import itertools
import lxml.etree as etree


def tokenize(line):
    def tokenize_word(w):
        words = []
        for _, group in itertools.groupby(w, key=lambda c: c.isalnum()):
            words.append(''.join(group))
        return words
    return [w for tok in line.split() for w in tokenize_word(tok)]


def load_milton(path='source/pl.xml'):
    lines = {}

    with open(path) as f:
        for book in etree.fromstring(f.read()).xpath('//div1[@type="Book"]'):
            book_id = int(book.attrib['n'])
            for l_id, l in enumerate(book.findall('l')):
                l_id += 1
                assert int(l.attrib.get('n', l_id)) == l_id, \
                    (l.attrib.get('n', l_id), l_id)
                text = ' '.join(tokenize(''.join(l.itertext()).strip()))
                assert (book_id, l_id) not in lines
                lines[book_id, l_id] = text

    return lines


def load_vondel(path='source/vond001luci01_01.xml'):
    lines = {}
    parser = etree.XMLParser(ns_clean=True, recover=True, encoding='utf-8')
    with open(path) as f:
        tree = etree.fromstring(f.read().encode(), parser=parser)
    for act_id, act in enumerate(tree.xpath('//div[@type="act"]')):
        act_id += 1
        for l_id, l in enumerate(act.xpath('.//l')):
            l_id += 1
            text = ' '.join(tokenize(''.join(l.itertext()).strip()))
            lines[act_id, l_id] = text

    return lines


lost = load_milton()
vondel = load_vondel()
