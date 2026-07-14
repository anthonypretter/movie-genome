#!/usr/bin/env python3
"""Export v2 site data: movies (with ratings) + cluster tree."""
import json, pickle, os
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
from export import NAMES  # reuse curated names

d = pickle.load(open(W + "/corpus.pkl", "rb"))
movies = d["movies"]
top = np.load(W + "/top.npy"); sub = np.load(W + "/sub.npy")
tree = json.load(open(W + "/tree.json"))
ratings = json.load(open(W + "/ratings.json"))

out_movies = []
for i, m in enumerate(movies):
    ex = m["e"]
    if len(ex) > 200: ex = ex[:197].rsplit(" ", 1)[0] + "…"
    out_movies.append([
        m["t"], m["y"] or 0, int(top[i]), int(sub[i]),
        m.get("th") or "", ex, m["href"] or "", ratings[i],
    ])

out_tree = []
for c in tree:
    subs = [{"name": " · ".join(s["terms"][:3]) or "misc", "n": s["n"]} for s in c["subs"]]
    out_tree.append({"id": c["id"], "name": NAMES[c["id"]],
                     "terms": c["terms"][:5], "n": c["n"], "subs": subs})

js = json.dumps({"movies": out_movies, "tree": out_tree}, separators=(",", ":"))
open(W + "/data_v2.json", "w").write(js)
print("v2 data", len(out_movies), "movies,", len(js), "bytes")
