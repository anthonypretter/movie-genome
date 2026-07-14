#!/usr/bin/env python3
"""Movie genome prototype pipeline. Run stages: corpus | embed | cluster | label | layout"""
import json, sys, glob, pickle, re, os
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))

# words that must never become genre labels: languages, nationalities, generic film words
BLOCKLIST = set("""
film films movie movies directed starring stars based story american british french japanese
italian german indian hindi tamil telugu malayalam kannada bengali marathi punjabi korean chinese
spanish mexican russian canadian australian english language soviet hong kong nigerian turkish
swedish norwegian danish dutch polish brazilian argentine iranian persian filipino thai vietnamese
egyptian pakistani indonesian greek czech hungarian romanian israeli finnish
america britain france japan italy germany india korea china spain mexico russia canada australia
released release remake sequel adaptation adapted novel play cast plays role lead
directorial debut written writer producer production stars co th feature length
produced screenplay million reviews received review festival premiered follows premiere
best award awards box office gross grossed budget opened distributed pictures studios
screen motion critics critical success commercial titled versus alongside features
john james michael david robert richard william george peter paul mary jack tom harry
sam charlie ben frank joe billy bobby johnny jimmy tony steve kevin brian gary larry
""".split())

def stage_corpus():
    movies = []
    for f in sorted(glob.glob(W + "/movies-*.json")):
        for m in json.load(open(f)):
            ext = m.get("extract") or ""
            if len(ext) < 200:
                continue
            movies.append({
                "t": m["title"], "y": m.get("year"),
                "g": m.get("genres") or [], "c": (m.get("cast") or [])[:6],
                "e": ext, "th": m.get("thumbnail"), "href": m.get("href"),
            })
    # dedupe by title+year
    seen, out = set(), []
    for m in movies:
        k = (m["t"].lower(), m["y"])
        if k in seen: continue
        seen.add(k); out.append(m)
    docs = []
    for m in out:
        # NOTE: deliberately exclude language/country fields from the doc
        docs.append(f"{m['t']}. Genres: {', '.join(m['g'])}. {m['e']} Cast: {', '.join(m['c'])}.")
    pickle.dump({"movies": out, "docs": docs}, open(W + "/corpus.pkl", "wb"))
    print("corpus", len(out))

def stage_embed():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    d = pickle.load(open(W + "/corpus.pkl", "rb"))
    vec = TfidfVectorizer(max_features=40000, stop_words="english",
                          ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(d["docs"])
    svd = TruncatedSVD(n_components=300, random_state=42)
    V = normalize(svd.fit_transform(X))
    np.save(W + "/vecs.npy", V.astype(np.float32))
    pickle.dump(vec, open(W + "/tfidf.pkl", "wb"))
    print("embedded", V.shape, "var", svd.explained_variance_ratio_.sum().round(3))

def stage_cluster():
    from sklearn.cluster import KMeans
    V = np.load(W + "/vecs.npy")
    K = 26
    km = KMeans(n_clusters=K, n_init=4, random_state=42)
    top = km.fit_predict(V)
    sub = np.zeros(len(V), dtype=int)
    for k in range(K):
        idx = np.where(top == k)[0]
        nk = max(2, min(9, len(idx) // 45))
        if len(idx) < 30:
            sub[idx] = 0; continue
        km2 = KMeans(n_clusters=nk, n_init=3, random_state=42)
        sub[idx] = km2.fit_predict(V[idx])
    np.save(W + "/top.npy", top); np.save(W + "/sub.npy", sub)
    print("clustered", K, "top clusters; sizes:", np.bincount(top).tolist())

def ctfidf_labels(doc_groups, n_terms=6):
    """c-TF-IDF: concat each group's docs, score terms frequent in group, rare overall."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=60000)
    X = cv.fit_transform(doc_groups)
    terms = np.array(cv.get_feature_names_out())
    tf = X.toarray().astype(float)
    tf = tf / (tf.sum(1, keepdims=True) + 1e-9)
    df = (X.toarray() > 0).sum(0)
    idf = np.log(1 + len(doc_groups) / (df + 1))
    S = tf * idf
    out = []
    for row in S:
        cand = terms[np.argsort(-row)]
        keep = []
        for t in cand:
            words = t.split()
            if any(w in BLOCKLIST for w in words): continue
            if any(len(w) < 3 for w in words): continue
            if re.search(r"\d", t): continue
            # skip terms fully contained in an already-kept longer term or vice versa
            if any(t in k or k in t for k in keep): continue
            keep.append(t)
            if len(keep) == n_terms: break
        out.append(keep)
    return out

def stage_label():
    d = pickle.load(open(W + "/corpus.pkl", "rb"))
    movies, docs = d["movies"], d["docs"]
    top = np.load(W + "/top.npy"); sub = np.load(W + "/sub.npy")
    K = top.max() + 1
    # strip title/cast from docs for labeling: label on plot+genres only
    ldocs = [f"{' '.join(m['g'])} . {m['e']}" for m in movies]
    top_groups = [" ".join(ldocs[i] for i in np.where(top == k)[0]) for k in range(K)]
    top_terms = ctfidf_labels(top_groups)
    tree = []
    for k in range(K):
        idx = np.where(top == k)[0]
        nsub = sub[idx].max() + 1
        sub_groups = [" ".join(ldocs[i] for i in idx[sub[idx] == s]) for s in range(nsub)]
        sub_terms = ctfidf_labels(sub_groups, n_terms=5)
        subs = []
        for s in range(nsub):
            sidx = idx[sub[idx] == s]
            subs.append({"terms": sub_terms[s], "n": int(len(sidx))})
        # dominant wiki genres as a hint
        from collections import Counter
        gc = Counter(g for i in idx for g in movies[i]["g"])
        tree.append({"id": int(k), "terms": top_terms[k], "n": int(len(idx)),
                     "wiki_genres": [g for g, _ in gc.most_common(3)], "subs": subs})
    json.dump(tree, open(W + "/tree.json", "w"), indent=1)
    for c in tree:
        print(c["id"], c["n"], c["wiki_genres"], "|", ", ".join(c["terms"][:4]))

def stage_layout():
    from sklearn.decomposition import PCA
    V = np.load(W + "/vecs.npy")
    top = np.load(W + "/top.npy")
    K = top.max() + 1
    cents = np.array([V[top == k].mean(0) for k in range(K)])
    C2 = PCA(2, random_state=42).fit_transform(cents)
    C2 = (C2 - C2.mean(0)) / (C2.std(0) + 1e-9)
    # spread centroids apart (simple repulsion)
    for _ in range(200):
        for a in range(K):
            f = np.zeros(2)
            for b in range(K):
                if a == b: continue
                dvec = C2[a] - C2[b]; dist = np.linalg.norm(dvec) + 1e-6
                if dist < 1.1: f += dvec / dist * (1.1 - dist) * 0.5
            C2[a] += f
    P = np.zeros((len(V), 2), dtype=np.float32)
    for k in range(K):
        idx = np.where(top == k)[0]
        local = PCA(2, random_state=0).fit_transform(V[idx])
        local = local / (np.abs(local).max() + 1e-9) * (0.32 + 0.05 * np.log10(len(idx)))
        P[idx] = C2[k] + local
    np.save(W + "/xy.npy", P)
    print("layout done", P.shape)

if __name__ == "__main__":
    {"corpus": stage_corpus, "embed": stage_embed, "cluster": stage_cluster,
     "label": stage_label, "layout": stage_layout}[sys.argv[1]]()
