import random
 
# create nodes
for i in range(50):
    g.addNode()
 
# create edges randomly
for u in g.nodes:
    for v in g.nodes:
        if random.random() < 0.5:
            g.addEdge(u, v)
 
# run force atlas layout
run_layout(ForceAtlas, iters=500)
