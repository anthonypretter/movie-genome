# Movie Genome

12,296 films embedded as vectors and clustered into 26 emergent genres + subgenres.
Browse by genre, search, sort by rating/year, or explore the cluster map.

**Live site:** rename repo URL here after enabling GitHub Pages
(Settings → Pages → Deploy from a branch → main, / root).

## How it was built
1. Movie plots + metadata (Wikipedia-sourced dataset, 1960s–2020s)
2. TF-IDF + SVD embeddings (300d) — swappable for gemini-embedding-001
3. KMeans clustering → 26 genres, ~7 subgenres each
4. c-TF-IDF auto-labeling with a blocklist (languages, countries, filler words)
5. Ratings joined from TMDB/IMDb dataset mirrors (~49% coverage)

Pipeline code in `/pipeline`. The site is a single self-contained `index.html`.
