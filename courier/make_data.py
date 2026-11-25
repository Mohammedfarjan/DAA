"""Generates sample datasets (data/small.json, data/large.json, data/medium.json)."""
import json, random

def make(path, n_nodes, n_parcels, capacity, seed, extra_edges):
    r = random.Random(seed)
    nodes = ["D"] + [f"C{i}" for i in range(1, n_nodes)]
    edges, seen = [], set()
    def add(u, v):
        k = tuple(sorted((u, v)))
        if u != v and k not in seen:
            seen.add(k); edges.append([u, v, r.randint(2, 20)])
    for i in range(1, n_nodes):                       # random spanning tree => connected
        add(nodes[i], nodes[r.randrange(i)])
    for _ in range(extra_edges):
        add(r.choice(nodes), r.choice(nodes))
    parcels = [dict(pid=f"P{i:03d}", dest=r.choice(nodes[1:]), weight=r.randint(1, 15),
                    priority=r.randint(1, 5), deadline=r.randint(2, 48)) for i in range(1, n_parcels + 1)]
    json.dump(dict(depot="D", capacity=capacity, nodes=nodes, edges=edges, parcels=parcels),
              open(path, "w"), indent=1)

# hand-made small dataset (easy to verify by hand)
small = dict(depot="D", capacity=20,
  nodes=["D","A","B","C","E","F"],
  edges=[["D","A",4],["D","B",2],["A","B",1],["A","C",5],["B","C",8],["B","E",10],
         ["C","E",2],["C","F",6],["E","F",3],["D","F",15]],
  parcels=[dict(pid="P01",dest="A",weight=5,priority=5,deadline=4),
           dict(pid="P02",dest="B",weight=8,priority=3,deadline=10),
           dict(pid="P03",dest="C",weight=6,priority=4,deadline=6),
           dict(pid="P04",dest="E",weight=9,priority=2,deadline=24),
           dict(pid="P05",dest="F",weight=4,priority=5,deadline=3),
           dict(pid="P06",dest="A",weight=7,priority=1,deadline=30),
           dict(pid="P07",dest="C",weight=3,priority=3,deadline=12),
           dict(pid="P08",dest="E",weight=10,priority=4,deadline=8)])
json.dump(small, open("data/small.json","w"), indent=1)
make("data/medium.json", 25, 40, 60, 7, 30)
make("data/large.json", 80, 150, 120, 42, 120)
print("datasets written")
