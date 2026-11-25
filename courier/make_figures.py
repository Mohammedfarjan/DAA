"""Creates architecture diagram and route diagram for the small dataset."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from courier_planner import *

# ---- architecture / flow diagram
fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
boxes = [("Input\nparcels.json\n+ road graph", 1), ("Merge Sort\n(priority /\ndeadline)", 21), ("0/1 Knapsack DP\n(select parcels\nfor one trip)", 41),
         ("Dijkstra\n(shortest paths\nbetween stops)", 61), ("Nearest Neighbour\nTSP (+2-opt)\n=> route", 81)]
for t, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 8), 17, 14, boxstyle="round,pad=0.4", fc="#dbeafe", ec="#1e3a8a"))
    ax.text(x + 8.5, 15, t, ha="center", va="center", fontsize=8)
for x in (18.6, 38.6, 58.6, 78.6):
    ax.annotate("", xy=(x + 2.4, 15), xytext=(x, 15), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(50, 3, "Timer wraps every step  ->  runtime report & charts", ha="center", fontsize=9, style="italic")
fig.savefig("outputs/fig_architecture.png", dpi=170, bbox_inches="tight"); plt.close()

# ---- route diagram
depot, cap, parcels, graph = load_dataset("data/small.json")
pos = {"D": (0, 0), "A": (2, 2), "B": (1, -1.5), "C": (4.5, 2.5), "E": (5, -1), "F": (7, 1)}
res = plan_trip(depot, cap, parcels, graph, use_2opt=True, verbose=False)
fig, ax = plt.subplots(figsize=(7, 4.2)); ax.axis("off")
seen = set()
for u in graph:
    for v, w in graph[u]:
        k = tuple(sorted((u, v)))
        if k in seen: continue
        seen.add(k)
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.plot([x1, x2], [y1, y2], color="#9ca3af", lw=1.2, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(w), fontsize=8, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
fp = res["full_path"]; 
for i, (a, b) in enumerate(zip(fp, fp[1:])):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    off = 0.08 if i >= len(fp) // 2 else -0.08
    ax.annotate("", xy=(x2, y2 + off), xytext=(x1, y1 + off), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8), zorder=3)
stops = {p.dest for p in res["chosen"]}
for n, (x, y) in pos.items():
    c = "#16a34a" if n == "D" else ("#f59e0b" if n in stops else "#e5e7eb")
    ax.scatter([x], [y], s=650, c=c, ec="black", zorder=4); ax.text(x, y, n, ha="center", va="center", fontweight="bold", zorder=5)
ax.set_title(f"Small dataset: delivery route (length {res['length']})\ngreen=depot, orange=delivery stop, grey=not needed, red arrows=road path", fontsize=9)
fig.savefig("outputs/fig_route_small.png", dpi=170, bbox_inches="tight"); plt.close()
print("figures ok", res["full_path"])
