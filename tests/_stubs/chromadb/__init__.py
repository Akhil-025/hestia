class Collection:
    def __init__(self, name, metadata=None):
        self.name = name; self.metadata = metadata
        self._ids = []; self._docs = []; self._metas = []; self._embs = []
    def count(self): return len(self._ids)
    def add(self, ids, documents, metadatas, embeddings):
        for i, d, m, e in zip(ids, documents, metadatas, embeddings):
            if i in self._ids: continue
            self._ids.append(i); self._docs.append(d)
            self._metas.append(m); self._embs.append(e)
    def get(self, where=None, include=None, limit=None):
        idxs = list(range(len(self._ids)))
        if where: idxs = [i for i in idxs if _match_where(self._metas[i], where)]
        if limit: idxs = idxs[:limit]
        result = {'ids': [self._ids[i] for i in idxs]}
        if include:
            if 'documents' in include: result['documents'] = [self._docs[i] for i in idxs]
            if 'metadatas' in include: result['metadatas'] = [self._metas[i] for i in idxs]
        return result
    def query(self, query_embeddings, n_results, where=None, include=None):
        idxs = list(range(len(self._ids)))
        if where: idxs = [i for i in idxs if _match_where(self._metas[i], where)]
        qe = query_embeddings[0]
        def dist(i):
            e = self._embs[i]
            return sum((a - b) ** 2 for a, b in zip(qe, e)) ** 0.5
        idxs.sort(key=dist); idxs = idxs[:n_results]
        return {'documents': [[self._docs[i] for i in idxs]],
                'metadatas': [[self._metas[i] for i in idxs]],
                'distances': [[dist(i) for i in idxs]]}
def _match_where(meta, where):
    if '$and' in where: return all(_match_where(meta, c) for c in where['$and'])
    return all(meta.get(k) == v for k, v in where.items())
class _Client:
    def __init__(self, path):
        self.path = path; self._collections = {}
    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = Collection(name, metadata)
        return self._collections[name]
    def delete_collection(self, name): self._collections.pop(name, None)
def PersistentClient(path): return _Client(path)
