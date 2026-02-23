"""
Skill Graph Traversal for QuantTradingOS.
Queries pgvector to find relevant skill nodes given a task context.
Injects skill context into agent calls via the orchestrator.
"""
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Add data-ingestion-service to path for db.connection
_svc = Path(__file__).resolve().parent.parent / "data-ingestion-service"
if str(_svc) not in sys.path:
    sys.path.insert(0, str(_svc))

import openai

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.75
RELATED_NODE_SIMILARITY_THRESHOLD = 0.65


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a query string."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.replace("\n", " "),
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def traverse_skill_graph(
    task_context: str,
    agent_name: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    include_shared: bool = True,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[dict]:
    """
    Given a task context, return the most relevant skill nodes.

    Parameters:
    - task_context: natural language description of current task
    - agent_name: if provided, prioritize nodes owned by this agent
    - top_k: number of nodes to return
    - include_shared: whether to include shared-skills nodes
    - min_similarity: minimum cosine similarity threshold

    Returns list of dicts: node_name, agent_name, node_path, content, similarity_score
    """
    from db.connection import get_connection

    embedding = get_embedding(task_context)
    embedding_str = str(embedding)

    conn = get_connection()
    cursor = conn.cursor()

    category_filter = ""
    if not include_shared:
        category_filter = "AND category != 'shared'"

    cursor.execute(f"""
        SELECT
            agent_name,
            node_name,
            node_path,
            category,
            content,
            related_nodes,
            1 - (embedding <=> %(embedding)s::vector) AS similarity_score
        FROM skill_nodes
        WHERE 1 - (embedding <=> %(embedding)s::vector) >= %(min_similarity)s
        {category_filter}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(top_k)s
    """, {
        "embedding": embedding_str,
        "min_similarity": min_similarity,
        "top_k": top_k * 2,
    })

    rows = cursor.fetchall()
    results = []

    for row in rows:
        score = float(row["similarity_score"])
        if agent_name and row["agent_name"] == agent_name:
            score = min(1.0, score * 1.15)
        results.append({
            "agent_name": row["agent_name"],
            "node_name": row["node_name"],
            "node_path": row["node_path"],
            "category": row["category"],
            "content": row["content"],
            "related_nodes": row["related_nodes"] or [],
            "similarity_score": score,
        })

    results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]

    related_paths = set()
    for result in results:
        related_paths.update(result["related_nodes"])

    if related_paths:
        cursor.execute("""
            SELECT agent_name, node_name, node_path, category, content, related_nodes,
                   1 - (embedding <=> %(embedding)s::vector) AS similarity_score
            FROM skill_nodes
            WHERE node_path = ANY(%(paths)s)
            AND 1 - (embedding <=> %(embedding)s::vector) >= %(threshold)s
        """, {
            "embedding": embedding_str,
            "paths": list(related_paths),
            "threshold": RELATED_NODE_SIMILARITY_THRESHOLD,
        })
        related_rows = cursor.fetchall()
        existing_paths = {r["node_path"] for r in results}
        for row in related_rows:
            if row["node_path"] not in existing_paths:
                results.append({
                    "agent_name": row["agent_name"],
                    "node_name": row["node_name"],
                    "node_path": row["node_path"],
                    "category": row["category"],
                    "content": row["content"],
                    "related_nodes": row["related_nodes"] or [],
                    "similarity_score": float(row["similarity_score"]),
                    "via_related_nodes": True,
                })

    cursor.close()
    conn.close()

    return results


def get_shared_context(regime: str | None = None, portfolio_state: dict | None = None) -> list[dict]:
    """
    Load shared context nodes: current_regime, portfolio_state, recent_decisions,
    and verified_performance nodes.
    """
    from db.connection import get_connection

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT agent_name, node_name, node_path, category, content, related_nodes
        FROM skill_nodes
        WHERE category = 'shared'
        ORDER BY node_name
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [{
        "agent_name": row["agent_name"],
        "node_name": row["node_name"],
        "node_path": row["node_path"],
        "category": row["category"],
        "content": row["content"],
        "related_nodes": row["related_nodes"] or [],
        "similarity_score": 1.0,
    } for row in rows]


def get_verified_performance(agent_name: str) -> dict:
    """
    Read the verified_performance node for a given agent from shared-skills/.
    Returns skill percentile and calibration info for dynamic weighting.
    """
    from db.connection import get_connection

    agent_map = {
        "Market-Regime-Agent": "regime_agent",
        "Sentiment-Shift-Alert-Agent": "sentiment_agent",
        "Equity-Insider-Intelligence-Agent": "insider_agent",
    }

    node_name = agent_map.get(agent_name)
    if not node_name:
        return {"skill_percentile": 0.5, "verified": False}

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT content FROM skill_nodes
        WHERE node_path LIKE %(path_pattern)s
        AND category = 'shared'
        LIMIT 1
    """, {"path_pattern": f"%verified_performance/{node_name}%"})

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return {"skill_percentile": 0.5, "verified": False}

    match = re.search(r"Skill Percentile:\s*([\d.]+)", row["content"])
    percentile = float(match.group(1)) if match else 0.5

    return {
        "skill_percentile": percentile,
        "verified": True,
        "content": row["content"],
    }


def format_skill_context(nodes: list[dict]) -> str:
    """
    Format retrieved skill nodes into a clean string for injection
    into an agent's context window.
    """
    if not nodes:
        return ""

    lines = ["--- SKILL CONTEXT ---"]
    for node in nodes:
        score = node.get("similarity_score", 0)
        via = " (via related nodes)" if node.get("via_related_nodes") else ""
        lines.append(f"\n### {node['node_name']} [{node['agent_name']}] (relevance: {score:.2f}){via}")
        lines.append(node["content"])
        lines.append("---")

    return "\n".join(lines)
