"""
Code Graph representation and AST-based parser for multi-relational code structures.
Extracts functions, classes, methods, calls, inheritance, and imports.
"""

import ast
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any


class NodeType(str, Enum):
    MODULE = "MODULE"
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    VARIABLE = "VARIABLE"


class EdgeType(str, Enum):
    CALLS = "CALLS"
    INHERITS = "INHERITS"
    IMPORTS = "IMPORTS"
    DEFINES = "DEFINES"
    REFERENCES = "REFERENCES"


@dataclass
class CodeNode:
    node_id: str
    name: str
    node_type: NodeType
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    signature: str = ""
    source_code: str = ""
    skeleton_code: str = ""
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Heuristic token count: ~4 characters per token."""
        return max(1, len(self.source_code) // 4)

    @property
    def skeleton_token_estimate(self) -> int:
        return max(1, len(self.skeleton_code) // 4)


@dataclass
class CodeEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeGraph:
    """
    Multi-relational directed code graph containing code symbol nodes
    and structural relationship edges.
    """

    def __init__(self, name: str = "RepositoryGraph"):
        self.name = name
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []
        self._adj_out: Dict[str, List[CodeEdge]] = {}
        self._adj_in: Dict[str, List[CodeEdge]] = {}
        self._name_to_ids: Dict[str, List[str]] = {}

    def add_node(self, node: CodeNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self._adj_out:
            self._adj_out[node.node_id] = []
        if node.node_id not in self._adj_in:
            self._adj_in[node.node_id] = []

        # Index by short symbol name for fast lookup
        if node.name not in self._name_to_ids:
            self._name_to_ids[node.name] = []
        self._name_to_ids[node.name].append(node.node_id)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            # Only connect nodes present in the graph
            return
        edge = CodeEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self.edges.append(edge)
        self._adj_out[source_id].append(edge)
        self._adj_in[target_id].append(edge)

    def get_out_edges(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[CodeEdge]:
        edges = self._adj_out.get(node_id, [])
        if edge_type is None:
            return edges
        return [e for e in edges if e.edge_type == edge_type]

    def get_in_edges(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[CodeEdge]:
        edges = self._adj_in.get(node_id, [])
        if edge_type is None:
            return edges
        return [e for e in edges if e.edge_type == edge_type]

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_type: Optional[EdgeType] = None,
    ) -> List[str]:
        neighbors: Set[str] = set()
        if direction in ("out", "both"):
            for e in self.get_out_edges(node_id, edge_type):
                neighbors.add(e.target_id)
        if direction in ("in", "both"):
            for e in self.get_in_edges(node_id, edge_type):
                neighbors.add(e.source_id)
        return list(neighbors)

    def lookup_by_name(self, name: str) -> List[CodeNode]:
        node_ids = self._name_to_ids.get(name, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def summary(self) -> Dict[str, Any]:
        edge_counts = {}
        for e in self.edges:
            edge_counts[e.edge_type.value] = edge_counts.get(e.edge_type.value, 0) + 1
        node_counts = {}
        for n in self.nodes.values():
            node_counts[n.node_type.value] = node_counts.get(n.node_type.value, 0) + 1
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_distribution": node_counts,
            "edge_distribution": edge_counts,
        }


class ASTGraphExtractor(ast.NodeVisitor):
    """
    Extracts AST symbol definitions and calls from a Python AST module.
    """

    def __init__(self, file_path: str, module_id: str, source_code: str):
        self.file_path = file_path
        self.module_id = module_id
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.graph = CodeGraph(name=module_id)
        self.current_parent_id: Optional[str] = None
        self.raw_calls: List[Tuple[str, str, int]] = []  # (caller_id, callee_name, lineno)

        # Add root module node
        self.root_module = CodeNode(
            node_id=module_id,
            name=module_id.split(".")[-1],
            node_type=NodeType.MODULE,
            file_path=file_path,
            start_line=1,
            end_line=len(self.lines) if self.lines else 1,
            docstring=ast.get_docstring(ast.parse(source_code)) if source_code.strip() else None,
            signature=f"module {module_id}",
            source_code="",
            skeleton_code=f"# Module: {module_id}",
        )
        self.graph.add_node(self.root_module)

    def _get_slice(self, start_line: int, end_line: int) -> str:
        s = max(0, start_line - 1)
        e = min(len(self.lines), end_line)
        return "\n".join(self.lines[s:e])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent_id = self.current_parent_id or self.module_id
        class_id = f"{parent_id}.{node.name}" if parent_id != self.module_id else f"{self.module_id}.{node.name}"
        doc = ast.get_docstring(node)
        source = self._get_slice(node.lineno, getattr(node, "end_lineno", node.lineno))

        bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else [b.id for b in node.bases if isinstance(b, ast.Name)]
        sig = f"class {node.name}({', '.join(bases)}):"
        skel = f"{sig}\n    \"\"\"{doc or ''}\"\"\"\n    ..."

        code_node = CodeNode(
            node_id=class_id,
            name=node.name,
            node_type=NodeType.CLASS,
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            docstring=doc,
            signature=sig,
            source_code=source,
            skeleton_code=skel,
            parent_id=parent_id,
            metadata={"bases": bases},
        )
        self.graph.add_node(code_node)
        self.graph.add_edge(parent_id, class_id, EdgeType.DEFINES, weight=1.0)

        # Record inheritance edges if base exists in graph (resolved later)
        for b_name in bases:
            self.raw_calls.append((class_id, b_name, node.lineno))

        old_parent = self.current_parent_id
        self.current_parent_id = class_id
        self.generic_visit(node)
        self.current_parent_id = old_parent

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node, is_async=True)

    def _handle_function(self, node: Any, is_async: bool) -> None:
        parent_id = self.current_parent_id or self.module_id
        func_id = f"{parent_id}.{node.name}"
        node_type = NodeType.METHOD if self.current_parent_id and self.graph.nodes.get(self.current_parent_id, None) and self.graph.nodes[self.current_parent_id].node_type == NodeType.CLASS else NodeType.FUNCTION

        doc = ast.get_docstring(node)
        source = self._get_slice(node.lineno, getattr(node, "end_lineno", node.lineno))

        # Format arguments
        args = []
        for a in node.args.args:
            arg_str = a.arg
            if a.annotation and hasattr(ast, "unparse"):
                arg_str += f": {ast.unparse(a.annotation)}"
            args.append(arg_str)

        ret = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) and hasattr(ast, "unparse") else ""
        prefix = "async def " if is_async else "def "
        sig = f"{prefix}{node.name}({', '.join(args)}){ret}:"
        doc_str = f'\n    """{doc}"""' if doc else ""
        skel = f"{sig}{doc_str}\n    ..."

        code_node = CodeNode(
            node_id=func_id,
            name=node.name,
            node_type=node_type,
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            docstring=doc,
            signature=sig,
            source_code=source,
            skeleton_code=skel,
            parent_id=parent_id,
            metadata={"args": args, "is_async": is_async},
        )
        self.graph.add_node(code_node)
        self.graph.add_edge(parent_id, func_id, EdgeType.DEFINES, weight=1.0)

        old_parent = self.current_parent_id
        self.current_parent_id = func_id
        self.generic_visit(node)
        self.current_parent_id = old_parent

    def visit_Call(self, node: ast.Call) -> None:
        caller_id = self.current_parent_id or self.module_id
        callee_name = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr
        if callee_name:
            self.raw_calls.append((caller_id, callee_name, node.lineno))
        self.generic_visit(node)


def build_code_graph_from_source(source_code: str, file_path: str = "main.py", module_id: str = "main") -> CodeGraph:
    """Builds a CodeGraph from raw python source string."""
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])

    extractor = ASTGraphExtractor(file_path=file_path, module_id=module_id, source_code=source_code)
    extractor.visit(tree)
    graph = extractor.graph

    # Resolve raw calls to edges
    for caller_id, callee_name, lineno in extractor.raw_calls:
        targets = graph.lookup_by_name(callee_name)
        for target in targets:
            edge_type = EdgeType.INHERITS if target.node_type == NodeType.CLASS and caller_id.endswith(callee_name) is False else EdgeType.CALLS
            graph.add_edge(caller_id, target.node_id, edge_type=edge_type, weight=1.0, metadata={"line": lineno})

    return graph


def build_code_graph_from_directory(repo_dir: str, max_files: int = 500) -> CodeGraph:
    """
    Recursively scans a repository directory and creates an interconnected multi-relational code graph.
    """
    root = Path(repo_dir)
    graph = CodeGraph(name=root.name)
    all_raw_calls: List[Tuple[str, str, int]] = []

    py_files = list(root.glob("**/*.py"))[:max_files]
    for p in py_files:
        try:
            rel_path = p.relative_to(root).as_posix()
            mod_id = rel_path.replace("/", ".").replace(".py", "")
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            sub_extractor = ASTGraphExtractor(file_path=rel_path, module_id=mod_id, source_code=content)
            try:
                tree = ast.parse(content, filename=rel_path)
                sub_extractor.visit(tree)
            except SyntaxError:
                continue

            for node in sub_extractor.graph.nodes.values():
                graph.add_node(node)
            for edge in sub_extractor.graph.edges:
                graph.edges.append(edge)
                graph._adj_out.setdefault(edge.source_id, []).append(edge)
                graph._adj_in.setdefault(edge.target_id, []).append(edge)
            all_raw_calls.extend(sub_extractor.raw_calls)
        except Exception:
            continue

    # Resolve cross-module calls
    for caller_id, callee_name, lineno in all_raw_calls:
        targets = graph.lookup_by_name(callee_name)
        for target in targets:
            graph.add_edge(caller_id, target.node_id, EdgeType.CALLS, weight=1.0, metadata={"line": lineno})

    return graph

