"""Memory for the language-model supervisor, after Park et al. (2023),
"Generative Agents: Interactive Simulacra of Human Behavior".

A memory stream is a list of natural-language records. Two kinds:
  observation  what happened after one decision: the situation, the choice,
               and the outcome over the following stretch of the run
  reflection   a higher-level insight the model distilled from observations

Retrieval, as in the paper, scores every record by
    recency    0.99 ** (decisions since it was written)
    importance 1-10, here computed from the outcome (collisions and soil
               damage are memorable, an uneventful stretch is not) instead of
               asking the model, which would double the number of calls
    relevance  cosine similarity between the record and the current
               situation, from an embedding model
each min-max normalised over the stream, summed with equal weights, and the
top k go into the prompt.

Reflection, as in the paper, is triggered when the summed importance of
observations since the last reflection passes a threshold: the model is
shown the most recent records and asked for its three most salient insights,
which are stored as reflections with high importance.

Embeddings come from Ollama (nomic-embed-text by default). HashEmbedder is a
dependency-free stand-in for tests.
"""
import json
import urllib.request
import zlib

import numpy as np


class OllamaEmbedder:
    def __init__(self, model='nomic-embed-text', host='http://127.0.0.1:11434', timeout=120):
        self.model, self.host, self.timeout = model, host.rstrip('/'), timeout

    def __call__(self, text):
        for path, body, key in (('/api/embed', dict(model=self.model, input=text), 'embeddings'),
                                ('/api/embeddings', dict(model=self.model, prompt=text), 'embedding')):
            try:
                req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    v = json.loads(r.read())[key]
                v = np.asarray(v[0] if key == 'embeddings' else v, float)
                return v / (np.linalg.norm(v) + 1e-12)
            except Exception as e:
                err = e
        raise RuntimeError('embedding failed: %s' % err)


class HashEmbedder:
    """Bag of words hashed into 256 dimensions. Crude, deterministic, offline."""
    def __call__(self, text):
        v = np.zeros(256)
        for w in str(text).lower().replace(',', ' ').replace('.', ' ').split():
            v[zlib.crc32(w.encode()) % 256] += 1.0
        return v / (np.linalg.norm(v) + 1e-12)


class MemoryStream:
    def __init__(self, embedder, reflectThreshold=40.0, decay=0.99):
        self.embed = embedder
        self.records = []          # dicts: text, kind, importance, step, vec
        self.step = 0              # decisions seen, the clock for recency
        self.sinceReflection = 0.0
        self.reflectThreshold = reflectThreshold
        self.decay = decay

    # ---- writing ----
    def add(self, text, importance, kind='observation'):
        self.records.append(dict(text=text, kind=kind, importance=float(importance),
                                 step=self.step, vec=self.embed(text)))
        if kind == 'observation':
            self.sinceReflection += importance

    def tick(self):
        self.step += 1

    def needsReflection(self):
        return self.sinceReflection >= self.reflectThreshold

    def reflect(self, llmChat, system, n=15):
        """Ask the model for 3 insights from the most recent records."""
        recent = self.records[-n:]
        if not recent:
            return []
        ask = ('Here are your most recent memories of leading the machines:\n' +
               '\n'.join('- ' + r['text'] for r in recent) +
               '\n\nWhat are the 3 most important, general insights you can infer from these about how '
               'to choose formation, speed and offset? Each should be one sentence that would help '
               'in a different field. Reply {"insights": ["...", "...", "..."]}')
        from .agents import _parseJSON
        out = _parseJSON(llmChat([dict(role='system', content=system), dict(role='user', content=ask)])) or {}
        ins = [str(x) for x in (out.get('insights') or [])][:3]
        for s in ins:
            self.add('Insight: ' + s, 8.0, kind='reflection')
        self.sinceReflection = 0.0
        return ins

    # ---- reading ----
    def retrieve(self, query, k=6):
        if not self.records:
            return []
        q = self.embed(query)
        rec = np.array([self.decay ** (self.step - r['step']) for r in self.records])
        imp = np.array([r['importance'] for r in self.records])
        rel = np.array([float(r['vec'] @ q) for r in self.records])

        def norm(x):
            span = x.max() - x.min()
            return (x - x.min()) / span if span > 1e-12 else np.zeros_like(x)
        score = norm(rec) + norm(imp) + norm(rel)
        idx = np.argsort(-score)[:k]
        return [self.records[i]['text'] for i in sorted(idx, key=lambda i: self.records[i]['step'])]

    # ---- persistence (vectors are recomputed on load) ----
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(dict(step=self.step, sinceReflection=self.sinceReflection,
                           records=[{k: v for k, v in r.items() if k != 'vec'} for r in self.records]), f, indent=1)

    @classmethod
    def load(cls, path, embedder, **kw):
        m = cls(embedder, **kw)
        try:
            with open(path) as f:
                d = json.load(f)
        except FileNotFoundError:
            return m
        m.step, m.sinceReflection = d['step'], d['sinceReflection']
        for r in d['records']:
            r['vec'] = embedder(r['text'])
            m.records.append(r)
        return m
