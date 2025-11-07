from bs4 import BeautifulSoup

pages = {
    'A': '''
    <html>
    <body>
    <a href="B">Go to B</a>
    <a href="C">Go to C</a>
    </body>
    </html>
    ''',

    'B': '''
    <html>
    <body>
    <a href="C">Go to C</a>
    <a href="A">Go to A</a>
    </body>
    </html>
    ''',

    'C': '''
    <html>
    <body>
    <a href="A">Go to A</a>
    </body>
    </html>
    ''',

    'D': '''
    <html>
    <body>
    <a href="C">Go to C</a>
    </body>
    </html>
    '''

}



graph = {}
for page, html in pages.items():
    soup = BeautifulSoup(html, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    graph[page] = [link for link in links if link in pages]  # Filter to internal links only

def pagerank(graph, d=0.85, tol=1e-6, max_iter=100):
    """
    Compute PageRank scores for the graph.
    
    Args:
    - graph: Dict of {node: [outgoing_neighbors]}
    - d: Damping factor (0 < d < 1)
    - tol: Convergence tolerance
    - max_iter: Max iterations
    
    Returns:
    - Dict of {node: PageRank score}
    """
    nodes = list(graph.keys())
    N = len(nodes)
    pr = {node: 1.0 / N for node in nodes}
    out_deg = {node: len(graph[node]) if len(graph[node]) > 0 else 1 for node in nodes}  # Avoid div by zero
    
    for _ in range(max_iter):
        new_pr = {node: (1 - d) / N for node in nodes}
        for node in nodes:
            for neighbor in graph[node]:
                new_pr[neighbor] += d * (pr[node] / out_deg[node])
        diff = sum(abs(new_pr[node] - pr[node]) for node in nodes)
        pr = new_pr
        if diff < tol:
            break
    
    return pr

ranks = pagerank(graph)

print("Graph (adjacency list):")
print(graph)
print("\nPageRank Scores:")

for node, score in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
    print(f"{node}: {score:.4f}")
