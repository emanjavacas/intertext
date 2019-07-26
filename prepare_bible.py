
import collections
import os

from lxml import etree


def get_doc_id(tree):
    return tree.find('//div1').attrib['n'] + '-' + tree.find('//div2').attrib['n']


def get_verses(tree):
    elems = tree.xpath('//*[local-name() = "s" or local-name() = "milestone"]')
    for idx, (milestone, s) in enumerate(zip(elems[::2], elems[1::2])):
        try:
            assert milestone.tag == 'milestone', milestone.tag
            assert s.tag == 's', s.tag
            assert idx + 1 == int(milestone.attrib['n']), (idx + 1, milestone.attrib['n'])
            yield idx + 1, ' '.join(s.text.split())
        except Exception as e:
            print(e)


def read_NT(path='bernard/source/NT/'):
    by_doc_id = collections.defaultdict(dict)
    for f in os.listdir(path):
        if not f.endswith('xml'):
            continue
        with open(os.path.join(path, f)) as fn:
            tree = etree.fromstring(fn.read().encode('utf-8')).getroottree()
        doc_id = get_doc_id(tree)
        for idx, verse in get_verses(tree):
            by_doc_id[doc_id][idx] = verse

    return by_doc_id


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='source/NT/')
    parser.add_argument('--target', default='output/NT.csv')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    import utils
    import pie
    model = pie.SimpleModel.load()
    model.to(args.device)

    with open(args.target, 'w') as f:
        for doc_id, verses in read_NT(path=args.source).items():
            book, chapter = doc_id.split('-')
            for verse_id, verse in verses.items():
                lemmas = utils.lemmatize(
                    model, verse.lower().split(), device='args.device')['lemma']
                lemmas = ' '.join(lemmas)
                f.write('\t'.join([book, chapter, str(verse_id), verse, lemmas]) + '\n')
