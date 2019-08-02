
import collections
import difflib

import prepare_bible


def LCS(a, b):
    return difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))


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


def read_bernard_scraped():
    import json

    with open('source/scraped.json') as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj['target'] is None:
                continue
            yield obj['url'].split('/')[-1], obj['target']


if __name__ == '__main__':
    targets = list(read_bernard_scraped())
    
    by_doc_id = prepare_bible.read_NT()
    
    dists = collections.defaultdict(lambda: collections.defaultdict(list))
    for doc_id, ids in by_doc_id.items():
        print(doc_id)
        for nt_idx, nt in ids.items():
            for bibl_id, bibl in targets:
                if abs(len(nt.split()) - len(bibl.split())) > 2:
                    continue
                dists[bibl_id][doc_id, nt_idx] = LCS(nt.split(), bibl.split()).size
    
    filtered = []
    for bibl_id in dists:
        for doc_id, nt_idx in dists[bibl_id]:
            if (dists[bibl_id][doc_id, nt_idx] /
                len(by_doc_id[doc_id][nt_idx].split()) > 0.9):
                filtered.append((doc_id, nt_idx, bibl_id))
    
    mappings = collections.defaultdict(set)
    for doc_id, nt_idx, bibl_id in filtered:
        _, stuff, book, verse_id = bibl_id.split('_')
        *_, stuff = stuff.split('%20')
        doc_id = doc_id.split('-')[0]
        if doc_id[0].isdigit():
            *_, doc_id = doc_id.split()
        mappings[stuff].add(doc_id)
    
    for stuff, doc_ids in mappings.items():
        if len(doc_ids) > 1:
            s = sorted(doc_ids, key=lambda doc_id: sum(c in stuff for c in doc_id), reverse=True)
            mappings[stuff] = s[0]
        else:
            mappings[stuff] = list(doc_ids)[0]
    
    with open('output/mappings.csv', 'w') as f:
        for bibl, nt in mappings.items():
            f.write(bibl + '\t' + nt + '\n')
