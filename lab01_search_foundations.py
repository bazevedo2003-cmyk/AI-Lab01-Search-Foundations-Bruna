from pathlib import Path

import numpy as np
import pandas as pd

from utils import bfs, dfs, ucs, haversine_km, load_openflights, plot_route_paths, route_is_valid

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab01"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------------- Question 1 ----------------
print("=== Question 1: data cleaning ===")
airports, routes, coords, graph, stats = load_openflights(
    DATA / "lab01-03_airports.csv", DATA / "lab01-03_routes.csv"
)
queries = pd.read_csv("utils/search_queries.csv").head(3)
print(pd.Series(stats, name="count").to_frame())
print(queries)

# ---------------- Question 2 ----------------
print("\n=== Question 2: distances ===")
sample = pd.DataFrame(graph["KUL"][:5], columns=["destination", "distance_km"])
print(sample)
direct_km = haversine_km(
    coords["KUL"]["latitude"], coords["KUL"]["longitude"],
    coords["KEF"]["latitude"], coords["KEF"]["longitude"],
)
print(f"Straight-line KUL -> KEF: {direct_km:.1f} km")

# ---------------- Question 3 ----------------
print("\n=== Question 3: BFS / DFS / UCS ===")
rows = []
for query in queries.itertuples():
    for name, algorithm in [("BFS", bfs), ("DFS", dfs), ("UCS", ucs)]:
        result = algorithm(graph, query.start, query.goal)
        rows.append({
            "query_id": query.query_id, "start": query.start, "goal": query.goal,
            "algorithm": name, **result.as_dict(), "path": " -> ".join(result.path),
        })
metrics = pd.DataFrame(rows).drop(columns="found")
metrics.to_csv(RESULTS / "metrics.csv", index=False)
print(metrics.drop(columns="path").round(2))
for row in metrics.itertuples():
    print(f"{row.query_id} {row.algorithm}: {row.path}")

# ---------------- Question 4 ----------------
print("\n=== Question 4: verification ===")
edge_km = {airport: dict(neighbours) for airport, neighbours in graph.items()}
for row in metrics.itertuples():
    path = row.path.split(" -> ")
    assert route_is_valid(path, graph), f"{row.algorithm} {row.query_id}: a leg is not a real route"
    recomputed = sum(edge_km[a][b] for a, b in zip(path, path[1:]))
    assert np.isclose(recomputed, row.path_cost_km), f"{row.algorithm} {row.query_id}: cost mismatch"
print("All 9 paths verified: every leg is a real route and every cost matches.")

# ---------------- Question 5 ----------------
print("\n=== Question 5: trade-offs ===")
print(metrics.pivot(index="query_id", columns="algorithm", values=["hops", "path_cost_km"]).round(0))
reversed_graph = {airport: neighbours[::-1] for airport, neighbours in graph.items()}
for query in queries.itertuples():
    original = dfs(graph, query.start, query.goal).hops
    reordered = dfs(reversed_graph, query.start, query.goal).hops
    print(f"{query.query_id} DFS: {original} hops with the original order, {reordered} with the reversed order")

# ---------------- Question 6 ----------------
print("\n=== Question 6: save evidence ===")
q1 = metrics[metrics.query_id == "Q1"]
paths = {row.algorithm: row.path.split(" -> ") for row in q1.itertuples()}
figure, _ = plot_route_paths(paths, coords, title="Route comparison: KUL to KEF")
figure.savefig(RESULTS / "route_comparison.png", dpi=160, bbox_inches="tight")
print("Lab 1 complete.")
print("Saved:", sorted(p.name for p in RESULTS.iterdir()))
