"""
Compatibility Adapter for the official Kaggle 'swegemma' package and dataset format.
Connects GraphSleuth algorithms to the competition's precomputed graphs and embeddings.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .graph_builder import CodeGraph, CodeNode, NodeType, EdgeType


class SWEGemmaAdapter:
    """
    Adapter to bridge Kaggle SWE-Gemma input formats (/kaggle/input/competitions/gemma-4-developer-agent)
    with GraphSleuth's multi-relational code graph representation.
    """

    def __init__(
        self,
        tasks_path: Optional[str] = None,
        graphs_dir: Optional[str] = None,
        embeddings_dir: Optional[str] = None,
    ):
        self.tasks_path = Path(tasks_path) if tasks_path else None
        self.graphs_dir = Path(graphs_dir) if graphs_dir else None
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Loads tasks from tasks.jsonl."""
        if not self.tasks_path or not self.tasks_path.exists():
            return []
        tasks = []
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    @staticmethod
    def from_networkx(nx_graph: Any, name: str = "ImportedGraph") -> CodeGraph:
        """
        Converts a NetworkX DiGraph (as returned by swegemma.graph.get_graph)
        into a GraphSleuth CodeGraph.
        """
        graph = CodeGraph(name=name)
        for node_id, data in nx_graph.nodes(data=True):
            ntype_str = data.get("type", "FUNCTION").upper()
            ntype = NodeType(ntype_str) if ntype_str in NodeType._value2member_map_ else NodeType.FUNCTION
            code_node = CodeNode(
                node_id=str(node_id),
                name=data.get("name", str(node_id).split(".")[-1]),
                node_type=ntype,
                file_path=data.get("file_path", "unknown.py"),
                start_line=data.get("start_line", 1),
                end_line=data.get("end_line", 1),
                docstring=data.get("docstring", None),
                signature=data.get("signature", f"def {node_id}():"),
                source_code=data.get("source_code", ""),
                skeleton_code=data.get("skeleton_code", f"def {node_id}():\n    ..."),
                parent_id=data.get("parent_id", None),
                metadata=data,
            )
            graph.add_node(code_node)

        for u, v, data in nx_graph.edges(data=True):
            etype_str = data.get("type", "CALLS").upper()
            etype = EdgeType(etype_str) if etype_str in EdgeType._value2member_map_ else EdgeType.CALLS
            weight = float(data.get("weight", 1.0))
            graph.add_edge(str(u), str(v), edge_type=etype, weight=weight, metadata=data)

        return graph

