"""
Baseline Implementations for Comparative Evaluation:
1. Dense Semantic RAG (Bi-Encoder / Embedding Cosine Similarity)
2. BM25 / Sparse Lexical Search
3. Naive k-Hop Graph Expansion
4. GraphSleuth (LTG + Multi-Relational PPR + HSGP)
"""

import math
import re
from typing import Dict, List, Set, Tuple
from graphsleuth.graph_builder import CodeGraph, CodeNode
from graphsleuth.ppr_engine import PersonalizedPageRankEngine, LexicalTopologicalGrounder
from graphsleuth.skeletonizer import HierarchicalSkeletonGraphPruner, PrunedContext


class DenseRAGBaseline:
    """
    Simulates dense semantic retrieval using character/token n-gram vectors
    to model precomputed vector embeddings (cosine similarity).
    """

    def __init__(self, graph: CodeGraph):
        self.graph = graph

    def _embed(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        counts: Dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm > 0:
            return {k: v / norm for k, v in counts.items()}
        return {}

    def _cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        score = 0.0
        for k, val in v1.items():
            if k in v2:
                score += val * v2[k]
        return score

    def rank(self, problem_statement: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_vec = self._embed(problem_statement)
        scores: List[Tuple[str, float]] = []

        for nid, node in self.graph.nodes.items():
            # Node embedding is generated from docstring + symbol name + signature
            doc_text = f"{node.name} {node.signature} {node.docstring or ''}"
            d_vec = self._embed(doc_text)
            sim = self._cosine(q_vec, d_vec)
            scores.append((nid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BM25LexicalBaseline:
    """
    Standard sparse BM25 lexical ranking across code symbols and source files.
    """

    def __init__(self, graph: CodeGraph, k1: float = 1.5, b: float = 0.75):
        self.graph = graph
        self.k1 = k1
        self.b = b
        self._index()

    def _index(self) -> None:
        self.docs: Dict[str, List[str]] = {}
        self.df: Dict[str, int] = {}
        self.avg_dl: float = 0.0

        total_len = 0
        for nid, node in self.graph.nodes.items():
            content = f"{node.name} {node.signature} {node.source_code}"
            toks = re.findall(r"[a-zA-Z0-9_]+", content.lower())
            self.docs[nid] = toks
            total_len += len(toks)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

        self.N = max(1, len(self.docs))
        self.avg_dl = total_len / self.N

    def rank(self, problem_statement: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_toks = re.findall(r"[a-zA-Z0-9_]+", problem_statement.lower())
        scores: List[Tuple[str, float]] = []

        for nid, doc_toks in self.docs.items():
            score = 0.0
            doc_len = len(doc_toks)
            tok_counts: Dict[str, int] = {}
            for t in doc_toks:
                tok_counts[t] = tok_counts.get(t, 0) + 1

            for t in q_toks:
                if t not in tok_counts:
                    continue
                tf = tok_counts[t]
                df = self.df.get(t, 0)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_dl)))
                score += idf * (numerator / denominator)

            scores.append((nid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class NaiveKHopBaseline:
    """
    Selects top lexical match and pulls all 2-hop neighbors with full unpruned source.
    """

    def __init__(self, graph: CodeGraph):
        self.graph = graph
        self.lexical = BM25LexicalBaseline(graph)

    def retrieve_context(self, problem_statement: str, k_hops: int = 2) -> Dict[str, Any]:
        ranked = self.lexical.rank(problem_statement, top_k=1)
        if not ranked:
            return {"nodes": [], "total_tokens": 0}

        seed_id = ranked[0][0]
        visited: Set[str] = {seed_id}
        frontier: Set[str] = {seed_id}

        for _ in range(k_hops):
            next_frontier = set()
            for nid in frontier:
                for neighbor in self.graph.get_neighbors(nid, direction="both"):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # Compute token volume (full unpruned source for all nodes)
        total_tokens = sum(
            self.graph.nodes[nid].token_estimate
            for nid in visited
            if nid in self.graph.nodes
        )
        return {
            "ranked_nodes": [(nid, 1.0) for nid in visited],
            "total_tokens": total_tokens,
            "num_nodes": len(visited),
        }

