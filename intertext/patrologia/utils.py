
RE_REF = r'([0-9]+-)?([A-Z]*[a-z]+-)*[A-Z][a-z]+_[0-9]+_[0-9]+'


def decode_ref(ref):
    book, chapter, verse = ref.split('_')
    book = ' '.join(book.split('-'))
    return book, chapter, verse


def encode_ref(ref):
    book, chapter, verse = ref
    return '-'.join(book.split()) + '_' + '_'.join([chapter, verse])
