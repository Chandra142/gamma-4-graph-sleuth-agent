"""
Automated Experiment Runner for GraphSleuth & Baselines.
Executes comparative evaluation and ablation studies, producing formatted scientific tables.
"""

import time
import json
import sys
import os
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Any
from graphsleuth.graph_builder import CodeGraph, build_code_graph_from_source
from graphsleuth.ppr_engine import PersonalizedPageRankEngine
from graphsleuth.skeletonizer import HierarchicalSkeletonGraphPruner
from evaluation.benchmark_dataset import get_curated_benchmark_tasks, BenchmarkTask
from evaluation.baselines import DenseRAGBaseline, BM25LexicalBaseline, NaiveKHopBaseline
from evaluation.metrics import compute_localization_metrics, compute_compression_metrics


def build_graph_for_task(task: BenchmarkTask) -> CodeGraph:
    """Combines all repo files in the task into a single unified CodeGraph."""
    graph = CodeGraph(name=task.repo)
    all_raw_calls = []

    for file_path, source in task.repo_files.items():
        sub_graph = build_code_graph_from_source(source, file_path=file_path, module_id=file_path.replace("/", ".").replace(".py", ""))
        for nid, node in sub_graph.nodes.items():
            graph.add_node(node)
        for edge in sub_graph.edges:
            graph.edges.append(edge)
            graph._adj_out.setdefault(edge.source_id, []).append(edge)
            graph._adj_in.setdefault(edge.target_id, []).append(edge)

    # Cross-module edge resolution
    for u in list(graph.nodes.keys()):
        for v in list(graph.nodes.keys()):
            if u != v:
                v_short = v.split(".")[-1]
                if v_short in graph.nodes[u].source_code:
                    from graphsleuth.graph_builder import EdgeType
                    graph.add_edge(u, v, EdgeType.CALLS, weight=1.0)

    return graph


def run_all_experiments() -> Dict[str, Any]:
    tasks = get_curated_benchmark_tasks()
    print(f"Loaded {len(tasks)} benchmark tasks across FastAPI, Requests, Rich, and HTTPX.\n")

    ground_truths = [t.gold_fault_nodes for t in tasks]

    # Containers for results
    results = {
        "Dense RAG": {"preds": [], "tokens": [], "time": 0.0},
        "BM25 Lexical": {"preds": [], "tokens": [], "time": 0.0},
        "Naive 2-Hop Expansion": {"preds": [], "tokens": [], "time": 0.0},
        "GraphSleuth (Ours)": {"preds": [], "tokens": [], "time": 0.0, "preserved_edges": [], "total_edges": []},
        "Ablation: No PPR (Lexical+HSGP)": {"preds": [], "tokens": [], "time": 0.0},
        "Ablation: No HSGP (PPR+FullDump)": {"preds": [], "tokens": [], "time": 0.0},
    }

    full_repo_tokens = []

    for task in tasks:
        graph = build_graph_for_task(task)
        repo_tok_sum = sum(n.token_estimate for n in graph.nodes.values())
        full_repo_tokens.append(repo_tok_sum)

        # 1. Dense RAG
        t0 = time.perf_counter()
        rag = DenseRAGBaseline(graph)
        rag_ranks = rag.rank(task.problem_statement, top_k=5)
        results["Dense RAG"]["time"] += (time.perf_counter() - t0)
        results["Dense RAG"]["preds"].append([r[0] for r in rag_ranks])
        # Dense RAG takes top-3 full nodes
        results["Dense RAG"]["tokens"].append(sum(graph.nodes[r[0]].token_estimate for r in rag_ranks[:3] if r[0] in graph.nodes))

        # 2. BM25 Lexical
        t0 = time.perf_counter()
        bm25 = BM25LexicalBaseline(graph)
        bm25_ranks = bm25.rank(task.problem_statement, top_k=5)
        results["BM25 Lexical"]["time"] += (time.perf_counter() - t0)
        results["BM25 Lexical"]["preds"].append([r[0] for r in bm25_ranks])
        results["BM25 Lexical"]["tokens"].append(sum(graph.nodes[r[0]].token_estimate for r in bm25_ranks[:3] if r[0] in graph.nodes))

        # 3. Naive 2-Hop
        t0 = time.perf_counter()
        nhop = NaiveKHopBaseline(graph)
        nhop_res = nhop.retrieve_context(task.problem_statement, k_hops=2)
        results["Naive 2-Hop Expansion"]["time"] += (time.perf_counter() - t0)
        results["Naive 2-Hop Expansion"]["preds"].append([r[0] for r in nhop_res["ranked_nodes"][:5]])
        results["Naive 2-Hop Expansion"]["tokens"].append(nhop_res["total_tokens"])

        # 4. GraphSleuth (Full Engine)
        t0 = time.perf_counter()
        ppr_engine = PersonalizedPageRankEngine(graph, alpha=0.18)
        ranked = ppr_engine.rank_focal_nodes(task.problem_statement, top_k=5)
        pruner = HierarchicalSkeletonGraphPruner(graph, max_focal_nodes=2, max_structural_nodes=4)
        pruned_ctx = pruner.prune(ranked)
        results["GraphSleuth (Ours)"]["time"] += (time.perf_counter() - t0)
        results["GraphSleuth (Ours)"]["preds"].append([r[0] for r in ranked])
        results["GraphSleuth (Ours)"]["tokens"].append(pruned_ctx.total_pruned_tokens)
        results["GraphSleuth (Ours)"]["preserved_edges"].append(pruned_ctx.preserved_edges_count)
        results["GraphSleuth (Ours)"]["total_edges"].append(max(1, pruned_ctx.total_relevant_edges))

        # 5. Ablation: No PPR (Lexical only + HSGP)
        t0 = time.perf_counter()
        no_ppr_ranked = bm25_ranks
        no_ppr_ctx = pruner.prune(no_ppr_ranked)
        results["Ablation: No PPR (Lexical+HSGP)"]["time"] += (time.perf_counter() - t0)
        results["Ablation: No PPR (Lexical+HSGP)"]["preds"].append([r[0] for r in no_ppr_ranked])
        results["Ablation: No PPR (Lexical+HSGP)"]["tokens"].append(no_ppr_ctx.total_pruned_tokens)

        # 6. Ablation: No HSGP (PPR + Full Dump)
        t0 = time.perf_counter()
        results["Ablation: No HSGP (PPR+FullDump)"]["time"] += (time.perf_counter() - t0)
        results["Ablation: No HSGP (PPR+FullDump)"]["preds"].append([r[0] for r in ranked])
        full_tokens = sum(graph.nodes[r[0]].token_estimate for r in ranked[:5] if r[0] in graph.nodes)
        results["Ablation: No HSGP (PPR+FullDump)"]["tokens"].append(full_tokens)

    # Compute comparative metrics table
    summary_table = []
    num_tasks = len(tasks)

    for method_name, data in results.items():
        loc_metrics = compute_localization_metrics(data["preds"], ground_truths)
        comp_metrics = compute_compression_metrics(full_repo_tokens, data["tokens"])
        avg_latency_ms = (data["time"] / num_tasks) * 1000.0

        row = {
            "Method": method_name,
            "Top-1 (%)": loc_metrics["top_1"],
            "Top-3 (%)": loc_metrics["top_3"],
            "Top-5 (%)": loc_metrics["top_5"],
            "MRR": loc_metrics["mrr"],
            "Avg Tokens": comp_metrics["avg_pruned_tokens"],
            "Compression (%)": comp_metrics["compression_percentage"],
            "Latency (ms)": round(avg_latency_ms, 2),
        }
        summary_table.append(row)

    return {
        "summary": summary_table,
        "full_repo_avg_tokens": round(sum(full_repo_tokens) / num_tasks, 1),
    }


def print_markdown_table(summary: List[Dict[str, Any]]) -> str:
    headers = ["Method", "Top-1 (%)", "Top-3 (%)", "Top-5 (%)", "MRR", "Avg Tokens", "Compression (%)", "Latency (ms)"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary:
        row_str = "| " + " | ".join(str(row[h]) for h in headers) + " |"
        lines.append(row_str)
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_all_experiments()
    print("=" * 80)
    print("EMPIRICAL EVALUATION RESULTS ACROSS SWE BENCHMARK TASKS:")
    print("=" * 80)
    print(print_markdown_table(res["summary"]))
    print("\nBenchmark completed successfully.")
