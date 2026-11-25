# Smart Courier Planning System - DAA Capstone

Name: Mohammed Farjan | Roll No.: 2501350080 | Course: B.Tech CSE (FSD) | Section: B
Subject: Design & Analysis of Algorithms | Submitted To: Anjali Pal

Pipeline: Merge Sort -> 0/1 Knapsack (DP) -> Dijkstra -> Nearest-Neighbour TSP (+2-opt)

## Run (Python 3.8+, only matplotlib needed for charts)
    pip install matplotlib
    python make_data.py          # create data/small.json, medium.json, large.json
    python courier_planner.py    # demo on small + large dataset
    python tests.py              # correctness checks (brute force / Floyd-Warshall)
    python benchmark.py          # runtime table + charts in outputs/
    python make_figures.py       # architecture + route diagrams
