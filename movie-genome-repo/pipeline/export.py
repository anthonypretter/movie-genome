#!/usr/bin/env python3
"""Export site data: movies + cluster tree with curated names."""
import json, pickle, os
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))

NAMES = {
 0: "Historical Epics", 1: "Romantic Dramas", 2: "Spy & Espionage",
 3: "Adventure & Fantasy Quests", 4: "Psychological Thrillers", 5: "Action Thrillers",
 6: "Serious Dramas", 7: "Comedy-Dramas & Coming of Age", 8: "Animation",
 9: "Noir & Neo-Noir", 10: "Studio Comedies", 11: "Westerns", 12: "Documentaries",
 13: "Mainstream & Family Comedies", 14: "Rom-Coms & Teen Romance",
 15: "Superhero & Martial Arts", 16: "Science Fiction", 17: "Slashers",
 18: "Murder Mysteries", 19: "Musicals", 20: "Monsters & Horror Franchises",
 21: "New York Stories", 22: "Crime & Gangsters", 23: "Indie & Festival Films",
 24: "Biopics & Sports Dramas", 25: "Supernatural Horror",
}

d = pickle.load(open(W + "/corpus.pkl", "rb"))
movies = d["movies"]
top = np.load(W + "/top.npy"); sub = np.load(W + "/sub.npy")
xy = np.load(W + "/xy.npy")
tree = json.load(open(W + "/tree.json"))

out_movies = []
for i, m in enumerate(movies):
    ex = m["e"]
    if len(ex) > 180: ex = ex[:177].rsplit(" ", 1)[0] + "…"
    out_movies.append([
        m["t"], m["y"] or 0, int(top[i]), int(sub[i]),
        round(float(xy[i][0]), 3), round(float(xy[i][1]), 3),
        m.get("th") or "", ex, m["href"] or "",
    ])

out_tree = []
for c in tree:
    subs = [{"name": " · ".join(s["terms"][:3]) or "misc", "n": s["n"]} for s in c["subs"]]
    out_tree.append({"id": c["id"], "name": NAMES[c["id"]],
                     "terms": c["terms"], "n": c["n"], "subs": subs})

data = {"movies": out_movies, "tree": out_tree}
js = json.dumps(data, separators=(",", ":"))
open(W + "/data.json", "w").write(js)
print("movies", len(out_movies), "bytes", len(js))
