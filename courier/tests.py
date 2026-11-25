"""Correctness evidence for every algorithm (run: python tests.py)."""
import random
from itertools import combinations
from courier_planner import *

def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok

def rand_parcels(n, r):
    return [Parcel(f"P{i}", f"C{r.randint(1,9)}", r.randint(1, 12), r.randint(1, 5), r.randint(1, 48)) for i in range(n)]

def floyd(graph):
    d = {u: {v: (0 if u == v else INF) for v in graph} for u in graph}
    for u in graph:
        for v, w in graph[u]: d[u][v] = min(d[u][v], w)
    for k in graph:
        for i in graph:
            for j in graph:
                if d[i][k] + d[k][j] < d[i][j]: d[i][j] = d[i][k] + d[k][j]
    return d

r = random.Random(1)
print("Sorting")
for n in (0, 1, 10, 200):
    ps = rand_parcels(n, r)
    check(f"merge_sort correct for n={n}", merge_sort(ps) == sorted(ps, key=sort_key))
ps = rand_parcels(60, r)
check("quick sort keys match", [sort_key(x) for x in quick_sort(ps)] == [sort_key(x) for x in merge_sort(ps)])
check("insertion sort keys match", [sort_key(x) for x in insertion_sort(ps)] == [sort_key(x) for x in merge_sort(ps)])

print("Knapsack (DP vs brute force over all subsets)")
for t in range(25):
    ps, cap = rand_parcels(10, r), r.randint(10, 50)
    best = max(sum(p.urgency_value for p in c) for k in range(11) for c in combinations(ps, k)
               if sum(p.weight for p in c) <= cap)
    val, chosen, _ = knapsack(ps, cap)
    assert val == best and sum(p.weight for p in chosen) <= cap and sum(p.urgency_value for p in chosen) == val
check("25 random instances: DP value == brute-force optimum, capacity respected", True)

print("Dijkstra (vs Floyd-Warshall)")
for f in ("data/small.json", "data/medium.json"):
    depot, cap, parcels, g = load_dataset(f)
    F = floyd(g)
    ok = all(dijkstra(g, s)[0][v] == F[s][v] for s in g for v in g)
    check(f"all-pairs distances match on {f}", ok)

print("TSP approximation (NN vs exact brute force, 7 stops)")
depot, cap, parcels, g = load_dataset("data/medium.json")
F = floyd(g)
ratios = []
for t in range(15):
    stops = r.sample([n for n in g if n != depot], 7)
    _, nn = nearest_neighbour(depot, stops, F)
    _, opt = brute_force_tsp(depot, stops, F)
    _, t2 = two_opt(nearest_neighbour(depot, stops, F)[0], F)
    assert nn >= opt and t2 >= opt and t2 <= nn
    ratios.append((nn / opt, t2 / opt))
check(f"NN never better than optimal; 2-opt never worse than NN", True)
print(f"  Average NN/OPT = {sum(a for a,_ in ratios)/len(ratios):.3f}, worst = {max(a for a,_ in ratios):.3f}")
print(f"  Average 2opt/OPT = {sum(b for _,b in ratios)/len(ratios):.3f}, worst = {max(b for _,b in ratios):.3f}")
print("ALL TESTS PASSED")
