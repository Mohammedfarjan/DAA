"""Runtime benchmarks + charts (run: python benchmark.py)."""
import random, time, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from courier_planner import *

def best_of(fn, reps=3):
    t = INF
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t = min(t, time.perf_counter() - t0)
    return t * 1000

def rp(n, r, nodes=50):
    return [Parcel(f"P{i}", f"C{r.randint(1,nodes)}", r.randint(1, 15), r.randint(1, 5), r.randint(1, 48)) for i in range(n)]

def rand_graph(n, r, extra):
    nodes = ["D"] + [f"C{i}" for i in range(1, n)]
    g = {x: [] for x in nodes}; seen = set()
    def add(u, v):
        k = tuple(sorted((u, v)))
        if u != v and k not in seen:
            seen.add(k); w = r.randint(2, 20); g[u].append((v, w)); g[v].append((u, w))
    for i in range(1, n): add(nodes[i], nodes[r.randrange(i)])
    for _ in range(extra): add(r.choice(nodes), r.choice(nodes))
    return g

r = random.Random(5)
results = {}

# 1. sorting
ns = [100, 500, 1000, 2000, 4000]
results["sort"] = {"n": ns, "merge": [], "quick": [], "insertion": [], "builtin": []}
for n in ns:
    ps = rp(n, r)
    results["sort"]["merge"].append(best_of(lambda: merge_sort(ps)))
    results["sort"]["quick"].append(best_of(lambda: quick_sort(ps)))
    results["sort"]["insertion"].append(best_of(lambda: insertion_sort(ps), 1))
    results["sort"]["builtin"].append(best_of(lambda: sorted(ps, key=sort_key)))

# 2. knapsack  (n x W)
ks = [25, 50, 100, 200, 400]
results["knap"] = {"n": ks, "W100": [], "W500": []}
for n in ks:
    ps = rp(n, r)
    results["knap"]["W100"].append(best_of(lambda: knapsack(ps, 100), 1))
    results["knap"]["W500"].append(best_of(lambda: knapsack(ps, 500), 1))

# 3. dijkstra
vs = [50, 200, 800, 3200, 6400]
results["dij"] = {"V": vs, "t": [], "E": []}
for v in vs:
    g = rand_graph(v, r, 2 * v)
    results["dij"]["E"].append(sum(len(x) for x in g.values()) // 2)
    results["dij"]["t"].append(best_of(lambda: dijkstra(g, "D")))

# 4. TSP NN vs 2-opt vs brute force
ss = [5, 7, 8, 9]
g = rand_graph(60, r, 100)
nodes = [n for n in g if n != "D"]
def apsp(sub):
    return {s: dijkstra(g, s)[0] for s in ["D"] + sub}
results["tsp_small"] = {"k": ss, "nn": [], "brute": []}
for k in ss:
    sub = r.sample(nodes, k); D = apsp(sub)
    results["tsp_small"]["nn"].append(best_of(lambda: nearest_neighbour("D", sub, D)))
    results["tsp_small"]["brute"].append(best_of(lambda: brute_force_tsp("D", sub, D), 1))
ks2 = [10, 20, 40, 60]
results["tsp_big"] = {"k": ks2, "nn": [], "two_opt": []}
for k in ks2:
    sub = r.sample(nodes, min(k, len(nodes))); D = apsp(sub)
    tour, _ = nearest_neighbour("D", sub, D)
    results["tsp_big"]["nn"].append(best_of(lambda: nearest_neighbour("D", sub, D)))
    results["tsp_big"]["two_opt"].append(best_of(lambda: two_opt(tour, D), 1))

# 5. full pipeline on 3 datasets
pipe = {}
for name in ("small", "medium", "large"):
    depot, cap, parcels, graph = load_dataset(f"data/{name}.json")
    runs = [plan_trip(depot, cap, parcels, graph, use_2opt=True, verbose=False) for _ in range(5)]
    avg = {k: sum(x["times"][k] for x in runs) / 5 for k in runs[0]["times"]}
    res = runs[0]
    pipe[name] = dict(times=avg, parcels=len(parcels), nodes=len(graph), loaded=len(res["chosen"]),
                      value=res["value"], nn=res["nn_length"], opt2=res["length"])
results["pipeline"] = pipe
json.dump(results, open("outputs/benchmark_results.json", "w"), indent=1)

# ---- charts
def line(ax, x, series, title, xl, log=False):
    for lab, y in series.items(): ax.plot(x, y, marker="o", label=lab)
    ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("time (ms)"); ax.grid(alpha=.3); ax.legend()
    if log: ax.set_yscale("log")

fig, ax = plt.subplots(figsize=(6, 4))
s = results["sort"]; line(ax, s["n"], {k: s[k] for k in ("insertion", "quick", "merge", "builtin")}, "Sorting runtime", "number of parcels (n)", True)
fig.tight_layout(); fig.savefig("outputs/chart_sorting.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
k = results["knap"]; line(ax, k["n"], {"capacity W=100": k["W100"], "capacity W=500": k["W500"]}, "DP Knapsack runtime  O(nW)", "number of parcels (n)")
fig.tight_layout(); fig.savefig("outputs/chart_knapsack.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
d = results["dij"]; line(ax, d["V"], {"Dijkstra (heap)": d["t"]}, "Dijkstra single-source runtime", "number of nodes (V), E ~ 3V")
fig.tight_layout(); fig.savefig("outputs/chart_dijkstra.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
t = results["tsp_small"]; line(axs[0], t["k"], {"Brute force (exact)": t["brute"], "Nearest Neighbour": t["nn"]}, "TSP: exact vs approximation", "stops", True)
t = results["tsp_big"]; line(axs[1], t["k"], {"Nearest Neighbour": t["nn"], "2-opt": t["two_opt"]}, "TSP: NN vs 2-opt (larger)", "stops", True)
fig.tight_layout(); fig.savefig("outputs/chart_tsp.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4)); import numpy as np
steps = ["sort", "knapsack", "dijkstra", "tsp_nn", "two_opt"]; w = .25
for i, nm in enumerate(pipe):
    ax.bar(np.arange(len(steps)) + i * w, [pipe[nm]["times"][s_] for s_ in steps], w, label=f"{nm} ({pipe[nm]['parcels']} parcels)")
ax.set_xticks(np.arange(len(steps)) + w); ax.set_xticklabels(steps); ax.set_yscale("log")
ax.set_ylabel("time (ms, log)"); ax.set_title("Pipeline step runtime by dataset"); ax.legend(); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig("outputs/chart_pipeline.png", dpi=150); plt.close()
print(json.dumps(results, indent=1))
