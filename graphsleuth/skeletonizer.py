"""
Hierarchical Skeleton Graph Pruner (HSGP).
Distills large repository code subgraphs into 3-tier semantic abstractions:
- Level 1 (Focal Implementation): Complete source code for probable fault locations.
- Level 2 (Structural Skeleton): AST signatures, type hints, docstrings with elided bodies.
- Level 3 (Reachability Stubs): Distal 1-line topological reachability signposts.
Reduces prompt context volume by 70-85% while preserving causal execution paths.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from .graph_builder import CodeGraph, CodeNode, NodeType


class SkeletonLevel(int, Enum):
    FOCAL_FULL = 1      # Full implementation code
    STRUCTURAL_AST = 2  # Signature + docstring + type annotations + elided body
    STUB_REACHABLE = 3  # One-line symbolic reachability stub


@dataclass
class PrunedNodeRepresentation:
    node_id: str
    level: SkeletonLevel
    file_path: str
    start_line: int
    end_line: int
    rendered_text: str
    original_tokens: int
    pruned_tokens: int

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return 1.0 - (self.pruned_tokens / max(1, self.original_tokens))


@dataclass
class PrunedContext:
    focal_node_ids: List[str]
    nodes: Dict[str, PrunedNodeRepresentation] = field(default_factory=dict)
    prompt_context_string: str = ""
    total_original_tokens: int = 0
    total_pruned_tokens: int = 0
    compression_percentage: float = 0.0
    preserved_edges_count: int = 0
    total_relevant_edges: int = 0


class HierarchicalSkeletonGraphPruner:
    """
    Transforms ranked PPR candidate nodes and their structural neighborhoods
    into a compressed prompt context window for Gemma 4.
    """

    def __init__(
        self,
        code_graph: CodeGraph,
        max_focal_nodes: int = 3,
        max_structural_nodes: int = 8,
        max_context_tokens: int = 4000,
    ):
        self.graph = code_graph
        self.max_focal = max_focal_nodes
        self.max_structural = max_structural_nodes
        self.max_context_tokens = max_context_tokens

    def prune(
        self,
        ranked_nodes: List[Tuple[str, float]],
    ) -> PrunedContext:
        """
        Takes PPR-ranked nodes [(node_id, score), ...] and produces a 3-tier pruned context.
        """
        focal_ids = [nid for nid, _ in ranked_nodes[:self.max_focal]]
        structural_candidates: Set[str] = set()

        # Gather 1-hop structural neighbors of focal nodes
        for fid in focal_ids:
            neighbors = self.graph.get_neighbors(fid, direction="both")
            for n in neighbors:
                if n not in focal_ids:
                    structural_candidates.add(n)

        # Also add next-ranked nodes from PPR if space permits
        for nid, _ in ranked_nodes[self.max_focal:]:
            if len(structural_candidates) >= self.max_structural:
                break
            structural_candidates.add(nid)

        structural_ids = list(structural_candidates)[:self.max_structural]

        # Gather 2-hop distal reachability stubs
        stub_candidates: Set[str] = set()
        for sid in structural_ids:
            for n in self.graph.get_neighbors(sid, direction="out"):
                if n not in focal_ids and n not in structural_ids:
                    stub_candidates.add(n)
        stub_ids = list(stub_candidates)[:10]

        # Build node representations
        pruned_nodes: Dict[str, PrunedNodeRepresentation] = {}
        total_orig = 0
        total_pruned = 0

        # Tier 1: Focal Full Bodies
        for nid in focal_ids:
            node = self.graph.nodes.get(nid)
            if not node:
                continue
            orig_tok = node.token_estimate
            rendered = (
                f"### [FOCAL FAULT CANDIDATE] {node.node_id}\n"
                f"File: {node.file_path} (Lines {node.start_line}-{node.end_line})\n"
                f"```python\n{node.source_code}\n```"
            )
            pruned_tok = max(1, len(rendered) // 4)
            pruned_nodes[nid] = PrunedNodeRepresentation(
                node_id=nid,
                level=SkeletonLevel.FOCAL_FULL,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                rendered_text=rendered,
                original_tokens=orig_tok,
                pruned_tokens=pruned_tok,
            )
            total_orig += orig_tok
            total_pruned += pruned_tok

        # Tier 2: Structural AST Skeletons
        for nid in structural_ids:
            node = self.graph.nodes.get(nid)
            if not node:
                continue
            orig_tok = node.token_estimate
            rendered = (
                f"### [DEPENDENCY SKELETON] {node.node_id}\n"
                f"File: {node.file_path} (Lines {node.start_line}-{node.end_line})\n"
                f"```python\n{node.skeleton_code}\n```"
            )
            pruned_tok = max(1, len(rendered) // 4)
            pruned_nodes[nid] = PrunedNodeRepresentation(
                node_id=nid,
                level=SkeletonLevel.STRUCTURAL_AST,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                rendered_text=rendered,
                original_tokens=orig_tok,
                pruned_tokens=pruned_tok,
            )
            total_orig += orig_tok
            total_pruned += pruned_tok

        # Tier 3: Reachable Stubs
        stub_lines = []
        for nid in stub_ids:
            node = self.graph.nodes.get(nid)
            if not node:
                continue
            orig_tok = node.token_estimate
            stub_line = f"- `{node.node_id}` ({node.node_type.value}) in {node.file_path}:{node.start_line}"
            stub_lines.append(stub_line)
            pruned_tok = max(1, len(stub_line) // 4)
            pruned_nodes[nid] = PrunedNodeRepresentation(
                node_id=nid,
                level=SkeletonLevel.STUB_REACHABLE,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                rendered_text=stub_line,
                original_tokens=orig_tok,
                pruned_tokens=pruned_tok,
            )
            total_orig += orig_tok
            total_pruned += pruned_tok

        # Assemble Markdown Context Prompt
        context_parts = [
            "## Repository Subgraph Context (Pruned via HSGP)\n",
            "### Tier 1: Target Implementations (Probable Bug Locations)\n",
        ]
        for nid in focal_ids:
            if nid in pruned_nodes:
                context_parts.append(pruned_nodes[nid].rendered_text + "\n")

        if structural_ids:
            context_parts.append("### Tier 2: Structural Dependencies (Signatures & Interfaces)\n")
            for nid in structural_ids:
                if nid in pruned_nodes:
                    context_parts.append(pruned_nodes[nid].rendered_text + "\n")

        if stub_lines:
            context_parts.append("### Tier 3: Distal Reachable Symbols\n")
            context_parts.append("\n".join(stub_lines) + "\n")

        assembled_prompt = "\n".join(context_parts)

        # Count preserved edges within the selected subgraph
        selected_set = set(pruned_nodes.keys())
        preserved_edges = 0
        total_subgraph_edges = 0
        for nid in selected_set:
            for e in self.graph.get_out_edges(nid):
                total_subgraph_edges += 1
                if e.target_id in selected_set:
                    preserved_edges += 1

        compression = 0.0
        if total_orig > 0:
            compression = max(0.0, (1.0 - (total_pruned / total_orig)) * 100.0)

        return PrunedContext(
            focal_node_ids=focal_ids,
            nodes=pruned_nodes,
            prompt_context_string=assembled_prompt,
            total_original_tokens=total_orig,
            total_pruned_tokens=total_pruned,
            compression_percentage=round(compression, 2),
            preserved_edges_count=preserved_edges,
            total_relevant_edges=total_subgraph_edges,
        )

