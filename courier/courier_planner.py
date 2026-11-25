"""
Smart Courier Planning System  (DAA Capstone)
=============================================
Integrates:
  1. Sorting              -> Merge Sort (stable, O(n log n)) to prioritise parcels
  2. Dynamic Programming  -> 0/1 Knapsack to pick best parcels for one trip
  3. Graph algorithms     -> Dijkstra (heap) for shortest paths between points
  4. Approximation        -> Nearest-Neighbour TSP (+ optional 2-opt) for the route
  5. Benchmarking         -> runtime of every major step

Run:  python courier_planner.py            (demo on small dataset + large dataset)
      python benchmark.py                  (runtime charts)
      python tests.py                      (correctness checks)
"""
import heapq
import json
import time
from dataclasses import dataclass
from itertools import permutations

INF = float("inf")


# ----------------------------------------------------------------------------
# DATA MODEL
# ----------------------------------------------------------------------------
@dataclass
class Parcel:
    pid: str
    dest: str          # delivery point (graph node)
    weight: int        # kg
    priority: int      # 1 (low) .. 5 (express)
    deadline: int      # hours from now

    @property
    def urgency_value(self):
        """Value used by knapsack: higher priority and tighter deadline => more valuable."""
        return self.priority * 10 + max(0, 24 - self.deadline)


def load_dataset(path):
    with open(path) as f:
        d = json.load(f)
    parcels = [Parcel(**p) for p in d["parcels"]]
    graph = {n: [] for n in d["nodes"]}
    for u, v, w in d["edges"]:               # undirected roads
        graph[u].append((v, w))
        graph[v].append((u, w))
    return d["depot"], d["capacity"], parcels, graph


# ----------------------------------------------------------------------------
# PHASE 2a - SORTING (Merge Sort) : key = (deadline asc, priority desc, weight asc)
# ----------------------------------------------------------------------------
def sort_key(p):
    return (p.deadline, -p.priority, p.weight)


def merge_sort(arr, key=sort_key):
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid], key), merge_sort(arr[mid:], key)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):     # <= keeps the sort stable
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


# Comparison sorts used only for benchmarking
def insertion_sort(arr, key=sort_key):
    a = arr[:]
    for i in range(1, len(a)):
        x, j = a[i], i - 1
        while j >= 0 and key(a[j]) > key(x):
            a[j + 1] = a[j]; j -= 1
        a[j + 1] = x
    return a


def quick_sort(arr, key=sort_key):
    if len(arr) <= 1:
        return arr[:]
    pivot = key(arr[len(arr) // 2])
    return (quick_sort([x for x in arr if key(x) < pivot], key)
            + [x for x in arr if key(x) == pivot]
            + quick_sort([x for x in arr if key(x) > pivot], key))


# ----------------------------------------------------------------------------
# PHASE 2b - DYNAMIC PROGRAMMING : 0/1 KNAPSACK
# ----------------------------------------------------------------------------
def knapsack(parcels, capacity):
    """Returns (best_value, selected_parcels, dp_table). O(n*W) time and space."""
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = parcels[i - 1].weight, parcels[i - 1].urgency_value
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c and dp[i - 1][c - w] + v > dp[i][c]:
                dp[i][c] = dp[i - 1][c - w] + v
    # backtrack
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(parcels[i - 1])
            c -= parcels[i - 1].weight
    chosen.reverse()
    return dp[n][capacity], chosen, dp


# ----------------------------------------------------------------------------
# PHASE 3a - DIJKSTRA
# ----------------------------------------------------------------------------
def dijkstra(graph, src):
    dist = {n: INF for n in graph}
    prev = {n: None for n in graph}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def path_to(prev, target):
    path = []
    while target is not None:
        path.append(target)
        target = prev[target]
    return path[::-1]


# ----------------------------------------------------------------------------
# PHASE 3b - TSP : NEAREST NEIGHBOUR (+ 2-opt improvement)
# ----------------------------------------------------------------------------
def nearest_neighbour(depot, stops, D):
    """D[a][b] = shortest-path distance. Returns tour [depot, ..., depot] and length."""
    unvisited, tour, cur, total = set(stops), [depot], depot, 0
    while unvisited:
        nxt = min(sorted(unvisited), key=lambda s: D[cur][s])   # sorted => deterministic tie-break
        total += D[cur][nxt]
        tour.append(nxt); unvisited.remove(nxt); cur = nxt
    total += D[cur][depot]
    tour.append(depot)
    return tour, total


def tour_length(tour, D):
    return sum(D[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


def two_opt(tour, D):
    """Optional optimisation: removes crossing edges until no improvement."""
    best = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if tour_length(new, D) < tour_length(best, D) - 1e-9:
                    best, improved = new, True
    return best, tour_length(best, D)


def brute_force_tsp(depot, stops, D):
    """Exact optimum - only for tiny inputs (<= 9 stops), used to measure NN quality."""
    best, best_len = None, INF
    for perm in permutations(stops):
        t = [depot, *perm, depot]
        L = tour_length(t, D)
        if L < best_len:
            best, best_len = t, L
    return best, best_len


# ----------------------------------------------------------------------------
# FULL PIPELINE
# ----------------------------------------------------------------------------
def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, (time.perf_counter() - t0) * 1000   # ms


def plan_trip(depot, capacity, parcels, graph, use_2opt=False, verbose=True):
    times = {}
    log = print if verbose else (lambda *a, **k: None)

    # 1. sort
    sorted_p, times["sort"] = timed(merge_sort, parcels)
    log("\n[1] PARCELS SORTED (deadline asc, priority desc)")
    for p in sorted_p:
        log(f"    {p.pid}  dest={p.dest:<3} wt={p.weight:<3} pr={p.priority} dl={p.deadline:<3}h value={p.urgency_value}")

    # 2. knapsack
    (best_val, chosen, dp), times["knapsack"] = timed(knapsack, sorted_p, capacity)
    left = [p for p in sorted_p if p not in chosen]
    log(f"\n[2] KNAPSACK  capacity={capacity}kg  best total value={best_val}")
    log("    Loaded : " + ", ".join(f"{p.pid}({p.weight}kg)" for p in chosen))
    log(f"    Weight used = {sum(p.weight for p in chosen)}kg   |  Left for next trip: " +
        (", ".join(p.pid for p in left) or "none"))

    # 3. dijkstra from depot + every stop
    stops = sorted({p.dest for p in chosen} - {depot})
    nodes_needed = [depot] + stops

    def all_pairs():
        D, prevs = {}, {}
        for s in nodes_needed:
            D[s], prevs[s] = dijkstra(graph, s)
        return D, prevs
    (D, prevs), times["dijkstra"] = timed(all_pairs)
    log(f"\n[3] DIJKSTRA from depot {depot}")
    for s in stops:
        log(f"    {depot} -> {s}: dist={D[depot][s]:<4} path={' -> '.join(path_to(prevs[depot], s))}")

    # 4. TSP
    (tour, length), times["tsp_nn"] = timed(nearest_neighbour, depot, stops, D)
    log(f"\n[4] NEAREST-NEIGHBOUR ROUTE  length={length}")
    log("    " + " -> ".join(tour))
    nn_len = length
    if use_2opt:
        (tour, length), times["two_opt"] = timed(two_opt, tour, D)
        log(f"    After 2-opt: length={length}  route: " + " -> ".join(tour))

    # expand to real road path
    full = [tour[0]]
    for a, b in zip(tour, tour[1:]):
        _, pv = dijkstra(graph, a)
        full += path_to(pv, b)[1:]
    log("    Full road path: " + " -> ".join(full))

    times["total"] = sum(times.values())
    log("\n[5] RUNTIME (ms): " + ", ".join(f"{k}={v:.3f}" for k, v in times.items()))
    return dict(sorted=sorted_p, chosen=chosen, left=left, value=best_val, tour=tour,
                length=length, nn_length=nn_len, full_path=full, times=times, dp=dp)


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["data/small.json", "data/large.json"]):
        print("=" * 78); print("DATASET:", name); print("=" * 78)
        depot, cap, parcels, graph = load_dataset(name)
        big = len(parcels) > 20
        res = plan_trip(depot, cap, parcels, graph, use_2opt=True, verbose=not big)
        if big:
            print(f"Parcels={len(parcels)} Nodes={len(graph)} Loaded={len(res['chosen'])} "
                  f"Value={res['value']} NN={res['nn_length']} 2opt={res['length']}")
            print("Runtime (ms):", {k: round(v, 3) for k, v in res['times'].items()})
