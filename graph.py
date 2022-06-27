import itertools
import random
import sys
from string import ascii_uppercase
from typing import List, Dict, Set
import enum
import uuid
import networkx as nx
import time
import matplotlib.pyplot as plt


class Edge:
    def __init__(self, node: 'Node', distance: int):
        self.id = str(uuid.uuid4())
        self.node = node
        self.distance = distance


class Node:
    def __init__(self, label: str):
        self.id = str(uuid.uuid4())
        self.label = label
        self.edges: List[Edge] = []

    # add another node as an edge
    def add_edge(self, edge_node: 'Node', distance: int) -> None:
        # prevent self referencing
        if self == edge_node:
            raise Exception("Node cannot be an edge to itself.")

        # prevent duplicate edges
        if edge_node in self.edges:
            raise Exception("Node already exists as an edge.")

        # add node to list of edges
        self.edges.append(Edge(edge_node, distance))

    def __hash__(self):
        return hash(str(self.id))

    def __eq__(self, other):
        if other is None or not isinstance(other, Node):
            return False

        return other.id == self.id


class PathType(enum.Enum):
    DIJKSTRA = 0
    BELLMAN_FORD = 1
    DEPTH_FIRST = 2


class Graph:
    def __init__(self):
        self.__nodes: List[Node] = []

    @property
    def nodes(self):
        return [x for x in self.__nodes]

    # get a node given a label
    def find(self, node: Node | str) -> Node | None:
        for x in self.__nodes:
            if x.label == node or x.id == node or x == node:
                return x
        return None

    # create a new node
    def add(self, label: str) -> Node:
        node = Node(label)

        # add new node with specified label
        self.__nodes.append(node)

        return node

    # connect one node to another
    def connect(self, a: Node | str, b: Node | str, distance: int = 1):
        a = self.find(a)
        b = self.find(b)

        # add edges
        a.add_edge(b, distance)
        b.add_edge(a, distance)

    # find and the shortest path between an origin and destination node (will render the result by default)
    def path(
        self, start: Node | str, end: Node | str,
        algorithm: PathType, display: bool = True
    ) -> [List[Node], float]:
        path = None
        start = self.find(start)
        end = self.find(end)

        # start performance benchmark
        start_time = time.time_ns()

        if algorithm == PathType.DIJKSTRA:
            path = self._dijkstra(start, end)
        elif algorithm == PathType.BELLMAN_FORD:
            path = self._bellman_ford(start, end)
        elif algorithm == PathType.DEPTH_FIRST:
            path = self._depth_first(start, end)

        # compute algorithm runtime
        runtime = time.time_ns() - start_time

        # render graph with path if display is true
        if display:
            self.display(path)

        return path, runtime

    def _dijkstra(self, start: Node, end: Node) -> List[Node]:
        unvisited: List[Node] = [x for x in self.__nodes]

        # node cost
        shortest_path = {}

        # shortest known path so far
        previous_nodes = {}

        # init all nodes to max value
        max_val = sys.maxsize
        for node in unvisited:
            shortest_path[node] = max_val

        # init starting node cost to o
        shortest_path[start] = 0

        # execute until all nodes are visited
        while unvisited:
            # find node with the lowest score
            current_min = None
            for node in unvisited:
                if current_min is None:
                    current_min = node
                elif shortest_path[node] < shortest_path[current_min]:
                    current_min = node

            # get the current nodes neighbors and update their distances
            neighbors: List[Edge] = current_min.edges
            for neighbor in neighbors:
                tentative_val = shortest_path[current_min] + neighbor.distance
                if tentative_val < shortest_path[neighbor.node]:
                    shortest_path[neighbor.node] = tentative_val
                    previous_nodes[neighbor.node] = current_min

            # after visiting neighbors mark node as visited
            unvisited.remove(current_min)

        # return a node path
        return self.build_path(start, end, previous_nodes)

    # bellman ford path
    def _bellman_ford(self, start: Node, end: Node) -> List[Node]:
        # distances and routes maps
        distances = {node: float("Inf") for node in self.__nodes}
        routes = {node: node for node in self.__nodes}

        # set distance of start node to 0
        distances[start] = 0

        for node in self.__nodes:
            for edge in node.edges:
                if distances[node] != float("Inf") and distances[node] + edge.distance < distances[edge.node]:
                    distances[edge.node] = distances[node] + edge.distance
                    routes[edge.node] = node

        # return a node path
        return self.build_path(start, end, routes)

    # depth first search for finding a path between two nodes
    def _depth_first(self, start: Node, end: Node, path: List[Node] = None, visited: Set[Node] = None):
        path = [] if path is None else path
        visited = set() if visited is None else visited
        path.append(start)
        visited.add(start)

        # return the path if we found the end node
        if start == end:
            return path

        # visit each edge for the current node if it has not yet been visited
        for edge in start.edges:
            if edge.node not in visited:
                result = self._depth_first(edge.node, end, path, visited)

                if result is not None:
                    return result

        # pop the current node from the  path and return none
        path.pop()
        return

    # build a path given a table of routes and a start and end node
    @staticmethod
    def build_path(start: Node, end: Node, routes: Dict[Node, Node]) -> List[Node]:
        # construct path
        path = []

        # follow backwards path to start node
        current = end
        while current != start:
            path.append(current)
            current = routes[current]

        # add last node
        path.append(current)

        # reverse and return path (so it goes from start -> end in the correct order)
        path.reverse()
        return path

    # display the graph for the user
    def display(self, path: List[Node] = None):
        # get start/end node if there is a path provided
        path_start = None if path is None else path[0]
        path_end = None if path is None else path[-1]

        # path edges
        path_edges = []

        # generate path edges if path is provided
        if path is not None:
            for i in range(0, len(path) - 1):
                path_edges.append((path[i], path[i + 1]))

        # graph
        graph = nx.Graph()

        # add nodes
        for node in self.__nodes:
            graph.add_node(node.id)

        # add edges
        for node in self.__nodes:
            for edge in node.edges:
                graph.add_edge(node.id, edge.node.id)

        # node color calculation function
        def node_color(n: Node):
            # if the node is the start node
            if n == path_start:
                return "#00ff00"
            # if the node is the end node
            elif n == path_end:
                return "#ff0000"
            # if hte node is traversed in the path
            elif path is not None and n in path:
                return "#ffff00"

            return "#1f78b4"

        # edge color calculation function
        def edge_color(n: Node, e: Edge):
            if (n, e.node) in path_edges or (e.node, n) in path_edges:
                return "#ffff00"

            return "k"

        # list of nodes and their edges
        node_edges = [(a, b) for a in self.__nodes for b in a.edges]

        # list of nodes
        nodes = [x for x in self.__nodes]
        node_ids = [x.id for x in nodes]
        node_colors = [node_color(x) for x in nodes]

        # list of edges
        edges = [x for x in node_edges]
        edge_ids = [(x[0].id, x[1].node.id) for x in edges]
        edge_colors = [edge_color(x[0], x[1]) for x in edges]

        # node label map
        labels = {node.id: node.label for node in self.__nodes}

        # if a path was defined then label the start and end nodes
        if path is not None:
            labels[path[0].id] = path[0].label + " (START)"
            labels[path[-1].id] = path[-1].label + " (END)"

        # generate node positions
        positions = nx.kamada_kawai_layout(graph)

        # edge label map
        edge_labels = {
            (x[0].id, x[1].node.id): x[1].distance
            for x in
            [(a, b) for a in self.__nodes for b in a.edges]
        }

        # get figsize given # of nodes (scales with * of nodes
        size = int(len(self.__nodes) / 5 * 2)

        # display graph
        plt.figure(3, figsize=(size, size))
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=node_ids,
            node_color=node_colors,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edge_ids,
            edge_color=edge_colors
        )
        nx.draw_networkx_labels(graph, positions, labels=labels)
        nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels)
        plt.show()


# generate alphabetical node names a, b, c.... aa, ab, ac...
def get_node_labels(count: int) -> List[str]:
    def iter_all_strings():
        for size in itertools.count(1):
            for s in itertools.product(ascii_uppercase, repeat=size):
                yield "".join(s)

    return [x for x in itertools.islice(iter_all_strings(), count)]


# create a random graph with a specified number of nodes and edge density
def get_random_graph(nodes: int = 25, density: int = 1) -> Graph:
    # create graph instance
    graph = Graph()

    # get node names
    nodes = get_node_labels(nodes)

    # add nodes
    for node in nodes:
        graph.add(node)

    # made at least one set of direct connections
    randomized_nodes = [x for x in nodes]
    random.shuffle(randomized_nodes)

    # connect nodes with a randomized weight
    for k in range(0, len(randomized_nodes) - 1):
        graph.connect(randomized_nodes[k], randomized_nodes[k + 1], random.randint(1, 10))

    # add connections based on density
    for i in range(0, density):
        # copy the nodes and randomize their order
        randomized_nodes = [x for x in nodes]
        random.shuffle(randomized_nodes)

        # connect nodes with a randomized weight
        for k in range(0, len(randomized_nodes)):
            # noinspection PyBroadException
            try:
                graph.connect(random.choice(randomized_nodes), random.choice(randomized_nodes), random.randint(1, 10))
            except Exception:
                pass
    return graph
