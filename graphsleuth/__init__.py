"""
GraphSleuth: Topological Code-Graph Distillation and Neuro-Symbolic Navigation
for Context-Constrained Local Developer Agents (Gemma 4).
"""

__version__ = "0.1.0"
__author__ = "Ram Chandra Gupta <ramchandargupta142@gmail.com>"

from .graph_builder import CodeGraph, CodeNode, CodeEdge, build_code_graph_from_source, build_code_graph_from_directory
from .ppr_engine import PersonalizedPageRankEngine, LexicalTopologicalGrounder
from .skeletonizer import HierarchicalSkeletonGraphPruner, SkeletonLevel, PrunedContext
from .agent_policy import GraphConstrainedAgent, AgentAction, AgentObservation
from .swegemma_adapter import SWEGemmaAdapter

__all__ = [
    "CodeGraph",
    "CodeNode",
    "CodeEdge",
    "build_code_graph_from_source",
    "build_code_graph_from_directory",
    "PersonalizedPageRankEngine",
    "LexicalTopologicalGrounder",
    "HierarchicalSkeletonGraphPruner",
    "SkeletonLevel",
    "PrunedContext",
    "GraphConstrainedAgent",
    "AgentAction",
    "AgentObservation",
    "SWEGemmaAdapter",
]

