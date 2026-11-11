# Vector search

A vector database stores embeddings, which are numeric representations of text. Similarity search finds the stored vectors that are closest to the vector of a query.

## Cosine similarity

Cosine similarity compares the angle between two vectors and ignores their length. When vectors are normalized to unit length, an inner product gives exactly the cosine similarity, and FAISS can compute it exactly over every stored vector.

## Rebuilding

An index is derived data. If the embedding model changes, the old vectors must be discarded and recomputed from the original text.
