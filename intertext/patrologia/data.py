
import regex as re

from intertext.patrologia import utils

RE_NUM = r"[0-9]+"


def load_patrologia(fpath, drop_punct=True, drop_num=True, lower=False, stopwords=None):
    tokens, refs = [], []
    with open(fpath) as f:
        for w in f:
            token, pos, lemma = w.strip().split('\t')
            # ref token
            if re.match(utils.RE_REF, token):
                refs.append((token, len(tokens)))
                continue
            # drop punctuation
            if drop_punct and (re.match(r'[\p{P}]+', token) or pos in ('PUN', 'SENT')):
                continue
            # drop stopwords based on lemma
            if stopwords is not None and lemma in stopwords:
                continue
            # drop number
            if drop_num and re.search(RE_NUM, token):
                continue
            # lower input token (not lemma)
            if lower:
                token = token.lower()

            tokens.append((token, pos, lemma))

    return tokens, refs
