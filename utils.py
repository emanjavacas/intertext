
import os
import pie


def lemmatize(model, sent, use_beam=True, beam_width=12, device='cpu'):
    inp, _ = pie.data.pack_batch(model.label_encoder, [sent], device=device)
    return model.predict(inp, "lemma", use_beam=use_beam, beam_width=beam_width)


def parse_ref(ref):
    _, book, chapter, verse = ref.split('_')
    *book_num, book = book.split('%20')
    book_num = book_num[0] if book_num else None
    return book_num, book, chapter, verse


def read_refs(path='bernard/output/bernard/refs.csv'):
    with open(path) as f:
        for line in f:
            line = line.strip()
            ref_type, ref, span = line.split('\t')
            span = set(idx for idx in span.split('-'))
            ref = parse_ref(ref)

            yield ref_type, ref, span


def read_mappings(path='bernard/output/mappings.csv'):
    with open(path) as f:
        mappings = {}
        for line in f:
            a, b = line.strip().split('\t')
            mappings[a] = b

    return mappings


def read_bernard_lines(path='bernard/output/bernard/docs'):
    for f in os.listdir(path):
        with open(os.path.join(path, f)) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line.split('\t')
                else:
                    yield None


def read_NT_lines(path='bernard/output/NT.csv'):
    by_id = {}
    with open(path) as f:
        for line in f:
            book, chapter, verse_num, verse = line.strip().split('\t')
            by_id[book, chapter, verse_num] = verse
    return by_id


def collect_refs():
    tokens, lemmas, poses = [], [], []
    id2idx = {}
    for line in read_bernard_lines():
        if line is None:
            continue

        bibl_id, tok, lem, pos = line
        assert bibl_id not in id2idx, "got known id {}".format(bibl_id)
        id2idx[bibl_id] = len(tokens)
        tokens.append(tok)
        lemmas.append(lem)
        poses.append(pos)

    missing = 0
    for ref_type, ref, span in read_refs():
        try:
            idxs = sorted(id2idx[idx] for idx in span)
            text = ' '.join([tokens[idx] for idx in idxs])

            yield ref_type, ref, span, text

        except KeyError:
            missing += 1

    print("Missing {} references".format(missing))


mappings = read_mappings()
NT = read_NT_lines()
refs = list(collect_refs())
missing = 0
output = []
for ref in refs:
    ref_type, (book_num, book, chapter, verse_num), span, text = ref
    try:
        NT_book = mappings[book]
        if book_num is not None:
            NT_book = book_num + ' ' + NT_book
        target = NT[NT_book, chapter, verse_num]
        output.append((ref, target))
    except Exception:
        missing += 1

# by_type = collections.defaultdict(list)
# for ref, target in output:
#     (ref_type, (book_num, book, chapter, verse_num), text) = ref
#     by_type[ref_type].append((ref, target))
