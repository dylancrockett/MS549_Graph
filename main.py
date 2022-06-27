from graph import Graph, PathType, get_random_graph
from consolemenu import *
from consolemenu.items import *

# main menu
menu = ConsoleMenu("Main Menu")

# graph
graph = get_random_graph()


# return the graph as a string
def get_graph():
    result = "Graph Nodes: \n["
    result += ", ".join([x.label for x in graph.nodes])
    result += "]"

    edges = [(node, x.node, x.distance) for node in graph.nodes for x in node.edges]
    unique_edges = []
    for (x, y, d) in edges:
        if not ((x, y) in unique_edges or (y, x) in unique_edges):
            unique_edges.append((x, y, d))

    result += "\n\nGraph Edges: \n["
    result += ", ".join([f"{x.label}-{y.label} ({d})" for (x, y, d) in unique_edges])
    result += "]"
    return result


# graph playground menu
graph_menu = ConsoleMenu("Graph Playground Menu", subtitle=get_graph)


# add node
def add_node():
    print("-" * 25)
    new_node = input("New Node Name: ")
    graph.add(new_node)
    print("Added new node successfully.")
    print("-" * 25)


# add edge
def add_edge():
    print("-" * 25)
    origin = input("Origin Node: ")
    destination = input("Origin Node: ")
    weight = int(input("Connection Weight: "))
    graph.connect(origin, destination, weight)
    print("Added new edge successfully.")
    print("-" * 25)


# clear graph
def clear_graph():
    global graph
    graph = Graph()


# find path between nodes
def find_path():
    # get node a and b
    def get_nodes():
        print("-" * 25)
        print("Available Nodes: [", ", ".join([x.label for x in graph.nodes]), "]")
        a, b = input("\nStart Node: "), input("\nEnd Node: ")
        print("-" * 25)
        return a, b

    # option map
    algo_map = {0: PathType.DIJKSTRA, 1: PathType.BELLMAN_FORD, 2: PathType.DEPTH_FIRST}
    algo = SelectionMenu.get_selection(["Dijkstra", "Bellman Ford", "Depth First"])

    # get the path type
    path_type = algo_map[algo]

    # get start and end node
    start, end = get_nodes()

    # find the path
    path, time_ns = graph.path(start, end, path_type)

    # print resulting path and time
    print("\n\n<> Results <>\n")
    print("Path Found: [", " -> ".join([x.label for x in path]), "]")
    print("Time to Find: ", time_ns)
    input("\nPress enter to continue...")


# display graph menu fnc
def display_graph():
    graph.display()


def load_scenario():
    global graph
    graph = Graph()
    nodes = ["Dylan Crockett", "Don Crockett", "Holly Crockett", "Ben Lomu", "Jessica Green", "Dan Ewing",
             "Micheal Rooker", "Kevin Bacon"]
    connections = [("Dylan Crockett", "Holly Crockett"), ("Dylan Crockett", "Don Crockett"),
                   ("Don Crockett", "Holly Crockett"), ("Dylan Crockett", "Ben Lomu"), ("Holly Crockett", "Ben Lomu"),
                   ("Don Crockett", "Ben Lomu"), ("Ben Lomu", "Jessica Green"), ("Jessica Green", "Dan Ewing"),
                   ("Dan Ewing", "Micheal Rooker"), ("Micheal Rooker", "Kevin Bacon")]

    for node in nodes:
        graph.add(node)

    for connection in connections:
        graph.connect(connection[0], connection[1])


# add graph menu options
graph_menu.append_item(FunctionItem("Add Node", add_node))
graph_menu.append_item(FunctionItem("Add Edge", add_edge))
graph_menu.append_item(FunctionItem("Clear Graph", clear_graph))
graph_menu.append_item(FunctionItem("Find Path", find_path))
graph_menu.append_item(FunctionItem("Display", display_graph))
graph_menu.append_item(FunctionItem("Load Custom Scenario", load_scenario))

# build main menu
menu.append_item(SubmenuItem("Graph Playground", graph_menu, menu))
menu.add_exit()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    menu.show()
