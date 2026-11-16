from math import inf

import pytest

from utils.core import astar, bfs, dfs, gbfs, ucs


def test_start_equals_goal():
    g = {"A": []}
    for algo in (bfs, dfs, ucs):
        r = algo(g, "A", "A")
        assert r.path == ["A"] and r.hops == 0 and r.path_cost_km == 0 and r.expanded == 0


def test_unreachable_goal():
    g = {"A": [("B", 1)], "B": [], "C": []}
    r = bfs(g, "A", "C")
    assert not r.found and r.path == [] and r.hops == -1 and r.path_cost_km == inf


def test_cycles_terminate():
    g = {"A": [("B", 1)], "B": [("A", 1), ("C", 1)], "C": []}
    assert dfs(g, "A", "C").found


def test_bfs_vs_ucs():
    g = {"S": [("G", 100), ("A", 1)], "A": [("B", 1)], "B": [("G", 1)], "G": []}
    assert bfs(g, "S", "G").path == ["S", "G"]
    assert ucs(g, "S", "G").path == ["S", "A", "B", "G"]


def test_ucs_updates_cheaper_path():
    g = {"S": [("A", 10), ("B", 1)], "A": [("G", 1)], "B": [("A", 1)], "G": []}
    r = ucs(g, "S", "G")
    assert r.path == ["S", "B", "A", "G"] and r.path_cost_km == 3


def test_astar_matches_ucs():
    g = {"S": [("A", 1), ("B", 4)], "A": [("G", 5)], "B": [("G", 1)], "G": []}
    h = {"S": 4, "A": 4, "B": 1, "G": 0}.get
    assert astar(g, "S", "G", h).path_cost_km == ucs(g, "S", "G").path_cost_km


def test_gbfs_not_optimal():
    g = {"S": [("A", 10), ("B", 1)], "A": [("G", 20)], "B": [("G", 1)], "G": []}
    h = {"S": 5, "A": 1, "B": 3, "G": 0}.get
    assert gbfs(g, "S", "G", h).path_cost_km > ucs(g, "S", "G").path_cost_km


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        ucs({"A": [("B", -1)], "B": []}, "A", "B")
