"""
Graph-Constrained Agent Policy for Gemma 4.
Enforces structural validity on tool calls and navigates the code graph
using ReAct reasoning loops without symbol hallucination.
"""

import ast
import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from .graph_builder import CodeGraph, EdgeType, NodeType
from .skeletonizer import HierarchicalSkeletonGraphPruner, PrunedContext


class ActionType(str, Enum):
    INSPECT_NODE = "inspect_node"
    EXPAND_NEIGHBORS = "expand_neighbors"
    DRAFT_PATCH = "draft_patch"
    RUN_SYNTAX_CHECK = "run_syntax_check"
    SUBMIT_FINAL_PATCH = "submit_final_patch"


@dataclass
class AgentAction:
    action_type: ActionType
    arguments: Dict[str, Any]
    thought: str = ""


@dataclass
class AgentObservation:
    success: bool
    data: Any
    message: str
    token_cost: int = 0


class GraphConstrainedAgent:
    """
    Local developer agent policy tailored for Gemma 4 (12B/4B/2B).
    Constrains tool execution to topologically valid operations on the CodeGraph.
    """

    def __init__(
        self,
        code_graph: CodeGraph,
        max_steps: int = 8,
        model_name: str = "gemma-4-12b-it",
    ):
        self.graph = code_graph
        self.max_steps = max_steps
        self.model_name = model_name
        self.history: List[Tuple[AgentAction, AgentObservation]] = []
        self.drafted_patch: Optional[str] = None

    def execute_action(self, action: AgentAction) -> AgentObservation:
        """
        Executes a single agent action with graph topology verification.
        """
        act_type = action.action_type
        args = action.arguments

        if act_type == ActionType.INSPECT_NODE:
            node_id = args.get("node_id", "")
            if node_id not in self.graph.nodes:
                # Graph constraint interception: symbol does not exist!
                similar = [
                    nid for nid in self.graph.nodes
                    if node_id.lower() in nid.lower() or nid.split(".")[-1] == node_id
                ][:3]
                return AgentObservation(
                    success=False,
                    data=None,
                    message=f"Constraint Error: Symbol '{node_id}' not found in repo code graph. Did you mean: {similar}?",
                    token_cost=30,
                )
            node = self.graph.nodes[node_id]
            data = {
                "node_id": node.node_id,
                "file_path": node.file_path,
                "lines": (node.start_line, node.end_line),
                "source": node.source_code,
                "docstring": node.docstring,
            }
            return AgentObservation(
                success=True,
                data=data,
                message=f"Retrieved node '{node_id}' ({node.node_type.value}) in {node.file_path}",
                token_cost=node.token_estimate,
            )

        elif act_type == ActionType.EXPAND_NEIGHBORS:
            node_id = args.get("node_id", "")
            edge_type_str = args.get("edge_type", None)
            edge_type = EdgeType(edge_type_str) if edge_type_str in EdgeType._value2member_map_ else None
            direction = args.get("direction", "both")

            if node_id not in self.graph.nodes:
                return AgentObservation(
                    success=False,
                    data=[],
                    message=f"Node '{node_id}' does not exist in graph.",
                    token_cost=20,
                )

            neighbors = self.graph.get_neighbors(node_id, direction=direction, edge_type=edge_type)
            neighbor_details = []
            for nid in neighbors[:8]:
                n = self.graph.nodes[nid]
                neighbor_details.append({
                    "node_id": n.node_id,
                    "type": n.node_type.value,
                    "signature": n.signature,
                    "file_path": n.file_path,
                })
            return AgentObservation(
                success=True,
                data=neighbor_details,
                message=f"Found {len(neighbors)} neighbors for '{node_id}'. Showing top {len(neighbor_details)}.",
                token_cost=sum(len(d["signature"]) // 4 for d in neighbor_details) + 40,
            )

        elif act_type == ActionType.DRAFT_PATCH:
            file_path = args.get("file_path", "")
            target_str = args.get("target", "")
            replacement_str = args.get("replacement", "")

            # Generate unified diff
            orig_lines = target_str.splitlines(keepends=True)
            new_lines = replacement_str.splitlines(keepends=True)
            diff = "".join(
                difflib.unified_diff(
                    orig_lines,
                    new_lines,
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                )
            )
            self.drafted_patch = diff
            return AgentObservation(
                success=True,
                data={"diff": diff, "file_path": file_path},
                message="Successfully formulated unified patch diff.",
                token_cost=max(20, len(diff) // 4),
            )

        elif act_type == ActionType.RUN_SYNTAX_CHECK:
            code_snippet = args.get("code", "")
            try:
                ast.parse(code_snippet)
                return AgentObservation(
                    success=True,
                    data={"valid_ast": True},
                    message="Python AST parse check passed without syntax errors.",
                    token_cost=15,
                )
            except SyntaxError as e:
                return AgentObservation(
                    success=False,
                    data={"valid_ast": False, "error": str(e), "lineno": e.lineno},
                    message=f"AST SyntaxError: {e.msg} at line {e.lineno}",
                    token_cost=25,
                )

        elif act_type == ActionType.SUBMIT_FINAL_PATCH:
            return AgentObservation(
                success=True,
                data={"final_patch": self.drafted_patch},
                message="Final patch submitted for benchmark evaluation.",
                token_cost=10,
            )

        return AgentObservation(
            success=False,
            data=None,
            message=f"Unknown action type: {act_type}",
            token_cost=10,
        )

    def generate_system_prompt(self, pruned_context: PrunedContext) -> str:
        """
        Creates an optimized system prompt for Gemma 4 incorporating the pruned context.
        """
        return (
            "You are an expert autonomous software engineer powered by Gemma 4.\n"
            "Your objective is to locate the root cause of a software issue and draft a precise, minimal patch.\n"
            "You have access to the topologically pruned repository context below.\n"
            "AVAILABLE ACTIONS: [inspect_node, expand_neighbors, draft_patch, run_syntax_check, submit_final_patch].\n"
            "STRICT CONSTRAINT: You may only navigate symbols that exist in the repository code graph.\n\n"
            f"{pruned_context.prompt_context_string}\n"
        )

