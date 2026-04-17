from datasketch import MinHash, MinHashLSH
import re

def preprocess(text):
    return re.findall(r'\w+', text.lower())

def build_lsh(texts, num_perm=128, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    for i, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for token in preprocess(text):
            m.update(token.encode('utf8'))
        lsh.insert(i, m)
        minhashes[i] = m
    return lsh, minhashes

def query_lsh(lsh, minhashes, text):
    m = MinHash(num_perm=128)
    for token in preprocess(text):
        m.update(token.encode('utf8'))
    return lsh.query(m)