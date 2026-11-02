"""
Personalized PageRank (PPR) Engine and Lexical-Topological Grounding for Code Graphs.
Solves the semantic-symbolic gap by anchoring problem statements to AST symbols
and diffusing relevance over multi-relational code dependency edges.
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple
from .graph_builder import CodeGraph, EdgeType, NodeType


class LexicalTopologicalGrounder:
    """
    Extracts candidate symbol anchors from natural language bug descriptions,
    tracebacks, and issue bodies, matching them against repository symbol nodes.
    """

    def __init__(self, code_graph: CodeGraph):
        self.graph = code_graph
        self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        self.symbol_tokens: Dict[str, Set[str]] = {}
        for nid, node in self.graph.nodes.items():
            tokens = set()
            # Tokenize node name (e.g. get_response -> [get, response])
            tokens.update(re.findall(r"[a-zA-Z0-9]+", node.name.lower()))
            # Tokenize node ID parts
            for part in nid.split("."):
                tokens.update(re.findall(r"[a-zA-Z0-9]+", part.lower()))
            self.symbol_tokens[nid] = tokens

    def extract_candidate_anchors(self, problem_statement: str) -> Dict[str, float]:
        """
        Extracts candidate anchor nodes and computes initial lexical match scores.
        """
        scores: Dict[str, float] = {}
        text_lower = problem_statement.lower()

        # 1. Exact identifier matches (e.g., function names, class names, dotted symbols)
        dotted_matches = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_.]+\b", problem_statement)
        backtick_matches = re.findall(r"`([a-zA-Z0-9_.]+)`", problem_statement)
        word_matches = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", problem_statement)
        traceback_lines = re.findall(r'File ".*?", line \d+, in ([a-zA-Z0-9_]+)', problem_statement)

        # High priority candidates
        high_priority = set(dotted_matches) | set(backtick_matches) | set(traceback_lines)

        for token in high_priority:
            # Check direct name lookup
            nodes = self.graph.lookup_by_name(token)
            for n in nodes:
                scores[n.node_id] = scores.get(n.node_id, 0.0) + 15.0

            # Check suffix / dotted containment in node_id
            tok_clean = token.strip("`")
            for nid, node in self.graph.nodes.items():
                if nid.endswith(tok_clean) or tok_clean in nid:
                    scores[nid] = scores.get(nid, 0.0) + 20.0

        # General identifier scoring via word overlap
        issue_words = set(w.lower() for w in word_matches)
        for nid, sym_tokens in self.symbol_tokens.items():
            overlap = sym_tokens.intersection(issue_words)
            if overlap:
                # Jaccard-like bonus
                match_score = sum(1.5 for tok in overlap if len(tok) > 3)
                if match_score > 0:
                    scores[nid] = scores.get(nid, 0.0) + match_score

        # Normalize scores to form a probability distribution p0
        if not scores:
            # Fallback uniform over all module/class definitions
            fallback_nodes = [
                nid for nid, n in self.graph.nodes.items()
                if n.node_type in (NodeType.CLASS, NodeType.FUNCTION)
            ]
            if not fallback_nodes:
                fallback_nodes = list(self.graph.nodes.keys())
            uniform_prob = 1.0 / max(1, len(fallback_nodes))
            return {nid: uniform_prob for nid in fallback_nodes}

        total_score = sum(scores.values())
        return {nid: score / total_score for nid, score in scores.items()}


class PersonalizedPageRankEngine:
    """
    Computes Multi-Relational Personalized PageRank (MR-PPR) via power iteration.
    p^(t+1) = (1 - alpha) * W^T * p^(t) + alpha * p_0
    """

    def __init__(
        self,
        code_graph: CodeGraph,
        alpha: float = 0.18,
        max_iter: int = 50,
        tol: float = 1e-6,
        edge_weights: Optional[Dict[EdgeType, float]] = None,
    ):
        self.graph = code_graph
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.edge_weights = edge_weights or {
            EdgeType.CALLS: 1.0,
            EdgeType.INHERITS: 0.85,
            EdgeType.DEFINES: 0.70,
            EdgeType.REFERENCES: 0.50,
            EdgeType.IMPORTS: 0.40,
        }
        self._build_transition_matrix()

    def _build_transition_matrix(self) -> None:
        """
        Pre-computes weighted normalized transition probabilities:
        M[u][v] = weight(u -> v) / sum_v'(weight(u -> v'))
        """
        self.transitions: Dict[str, List[Tuple[str, float]]] = {}
        for u in self.graph.nodes:
            out_edges = self.graph.get_out_edges(u)
            in_edges = self.graph.get_in_edges(u)  # Also allow bidirectional diffusion
            
            # Weighted neighbors: forward out-edges + 0.3 * backward in-edges
            neighbor_weights: Dict[str, float] = {}
            for e in out_edges:
                w = self.edge_weights.get(e.edge_type, 1.0) * e.weight
                neighbor_weights[e.target_id] = neighbor_weights.get(e.target_id, 0.0) + w

            for e in in_edges:
                w = self.edge_weights.get(e.edge_type, 1.0) * e.weight * 0.3
                neighbor_weights[e.source_id] = neighbor_weights.get(e.source_id, 0.0) + w

            total_weight = sum(neighbor_weights.values())
            if total_weight > 0:
                self.transitions[u] = [
                    (v, w / total_weight) for v, w in neighbor_weights.items()
                ]
            else:
                self.transitions[u] = []

    def compute_ppr(self, seed_distribution: Dict[str, float]) -> Dict[str, float]:
        """
        Runs power-iteration to compute stationary Personalized PageRank vector.
        """
        nodes = list(self.graph.nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}

        # Ensure seed distribution sums to 1.0
        p0 = {nid: 0.0 for nid in nodes}
        for nid, val in seed_distribution.items():
            if nid in p0:
                p0[nid] = val
        norm = sum(p0.values())
        if norm > 0:
            p0 = {k: v / norm for k, v in p0.items()}
        else:
            p0 = {nid: 1.0 / n for nid in nodes}

        # Initialize p with p0
        p = dict(p0)

        for iteration in range(self.max_iter):
            p_next = {nid: self.alpha * p0[nid] for nid in nodes}
            dangling_mass = 0.0

            for u, prob in p.items():
                if prob == 0.0:
                    continue
                trans = self.transitions.get(u, [])
                if not trans:
                    dangling_mass += prob
                else:
                    flow = (1.0 - self.alpha) * prob
                    for v, trans_prob in trans:
                        p_next[v] += flow * trans_prob

            # Redistribute dangling mass according to p0
            if dangling_mass > 0:
                dangling_flow = (1.0 - self.alpha) * dangling_mass
                for nid in nodes:
                    p_next[nid] += dangling_flow * p0[nid]

            # Check convergence L1 error
            diff = sum(abs(p_next[nid] - p[nid]) for nid in nodes)
            p = p_next
            if diff < self.tol:
                break

        return p

    def rank_focal_nodes(
        self,
        problem_statement: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        End-to-end ranking from natural-language issue to top-k localized focal nodes.
        Calibrates scores by node type: executable units (functions/methods) receive higher priority.
        """
        grounder = LexicalTopologicalGrounder(self.graph)
        seed_dist = grounder.extract_candidate_anchors(problem_statement)
        ppr_scores = self.compute_ppr(seed_dist)

        type_weights = {
            NodeType.METHOD: 1.25,
            NodeType.FUNCTION: 1.25,
            NodeType.CLASS: 0.90,
            NodeType.MODULE: 0.30,
            NodeType.VARIABLE: 0.40,
        }

        calibrated_scores = {}
        for nid, score in ppr_scores.items():
            node = self.graph.nodes.get(nid)
            multiplier = type_weights.get(node.node_type, 1.0) if node else 1.0
            calibrated_scores[nid] = score * multiplier

        sorted_nodes = sorted(calibrated_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_nodes[:top_k]
