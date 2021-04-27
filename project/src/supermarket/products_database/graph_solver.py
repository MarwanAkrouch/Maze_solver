"""The tools for problem solving with Integer Linear Programming"""

from __future__ import print_function
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# -------------Tools--------------


def vertices_from_graph(G):
    """
    Generate the set of the vertices from a graph

    Parameters
    -------
    G : dict

    Returns
    -------
    vertices_list : list

    """
    vertices_list = []
    for vertex in G:
        vertices_list.append(vertex)
    return vertices_list


def edges_from_graph(G):
    """
    Generate the set of the edges from a graph

    Parameters
    -------
    G : dict

    Returns
    -------
    edges_list : list

    """
    edges_list = []
    for vertex in G:
        for neighbor in G[vertex]:
            new_edge_1 = vertex + '-' + neighbor
            new_edge_2 = neighbor + '-' + vertex
            if not new_edge_1 in edges_list:
                edges_list.append(new_edge_1)
            if not new_edge_2 in edges_list:
                edges_list.append(new_edge_2)
    return edges_list


def contains_vertex(edge, vertex):
    """
    Determine if edge contains the vertex

    Parameters
    -------
    edge : str
    vertex : str

    Returns
    -------
    _ : bool

    """
    vertices = edge.split('-')
    return vertices[0] == vertex or vertices[1] == vertex


def contains_vertex_first(edge, vertex):
    """
    Determine if edge contains the vertex at the first position

    Parameters
    -------
    edge : str
    vertex : str

    Returns
    -------
    _ : bool

    """
    vertices = edge.split('-')
    return vertices[0] == vertex


def contains_vertex_second(edge, vertex):
    """
    Determine if edge contains the vertex at the second position

    Parameters
    -------
    edge : str
    vertex : str

    Returns
    -------
    _ : bool

    """
    vertices = edge.split('-')
    return vertices[1] == vertex

# ------------------------------------------


class Model:
    """
    A complete model needed to be solved.
    Convention : the vertex 's' is added to the start, the vertex 't' to the end.

    Attributes
    -------
    start : str
            the vertex where to start
    end : str
            the vertex where to end
    G : dict
            the graph of the initial problem
    E : list
            the set of the edges
    V : list
            the set of the vertices
    U : list
            the set of the vertices that need a visit
    nbr_edges : int
            the numbers of edges in E
    nbr_vertices : int
            the numbers of vertices in V
    """

    def __init__(self, graph, starting, ending, vertices_with_products):
        self.start = starting
        self.end = ending
        # -
        self.G = graph
        # -
        self.E = edges_from_graph(graph)
        self.E.append('s-' + starting)
        self.E.append(starting + '-s')
        self.E.append(ending + '-t')
        self.E.append('t-' + ending)
        # -
        self.V = vertices_from_graph(graph)
        self.V.append('s')
        self.V.append('t')
        # -
        self.U = vertices_with_products
        # -
        self.nbr_edges = len(self.E)
        self.nbr_vertices = len(self.V)


def shortest_path(model):
    """
    Solve the shortest path of a model using Integer Linear Programming

    Parameters
    -------
    model : Model (new class)
        the model to solve

    Returns
    -------
    solution : list
        list of the edges of the path
    cost : int
        cost of the path
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    infinity = solver.infinity()
    # -----Decision variables-----
    x = {}
    for j in range(model.nbr_edges):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)
    #print('Number of variables =', solver.NumVariables())
    # -----Constraints-----
    for i in model.V:
        where_i_first = []
        where_i_second = []
        for e in model.E:
            if contains_vertex_first(e, i):
                where_i_first.append(model.E.index(e))
            if contains_vertex_second(e, i):
                where_i_second.append(model.E.index(e))
        if i == 's':
            solver.Add(sum(x[e] for e in where_i_first) ==
                       1 + sum(x[e] for e in where_i_second))
        elif i == 't':
            solver.Add(sum(x[e] for e in where_i_first) ==
                       (-1) + sum(x[e] for e in where_i_second))
        else:
            solver.Add(sum(x[e] for e in where_i_first) ==
                       sum(x[e] for e in where_i_second))
    #print('Number of constraints =', solver.NumConstraints())
    # -----Objective-----
    solver.Minimize(sum(x[e] for e in range(model.nbr_edges)))
    status = solver.Solve()
    # -----Display-----
    if status == pywraplp.Solver.OPTIMAL:
        solution = []
        cost = 0
        for j in range(model.nbr_edges):
            if x[j].solution_value() > 0:
                solution.append(model.E[j])
                cost += 1
        #print('Problem solved in %f milliseconds' % solver.wall_time())
        return solution, cost
    else:
        print('The problem of the SHORTEST PATH does not have an optimal solution')
        return [], 0

# ------------------------FINAL SOLVER------------------------------


def create_data_model(model):
    """
    Get the data ready for the final solver.
    The conventions of the routing problem are kept for analogy and comprehension purposes.

    Parameters
    -------
    model : Model (new class)
        the complete model to solve

    Returns
    -------
    data : dict
        the whole data needed for the problem
    """
    W = []
    for u in model.U:
        W.append(u)
    W.append(model.start)
    W.append(model.end)

    data = {}
    len_W = len(W)
    matrix = np.zeros((len_W, len_W), dtype=int)
    data['path'] = {}

    for p in range(len_W):
        for q in range(p + 1, len_W):
            the_model = Model(model.G, W[p], W[q], W)
            the_solution = shortest_path(the_model)
            matrix[p, q] = the_solution[1]
            matrix[q, p] = the_solution[1]
            path_name = W[p] + '-' + W[q]
            data['path'][path_name] = the_solution[0]
    data['distance_matrix'] = matrix.tolist()
    data['num_vehicles'] = 1
    data['starts'] = [len_W - 2]
    data['ends'] = [len_W - 1]

    return data


def display_solution(manager, routing, solution):
    """
    Print the solution thanks to the Google library tools.
    The vocabulary of the routing problem is kept for analogy and comprehension purposes.

    Parameters
    -------
    data : dict
    manager : ortools.constraint_solver.pywrapcp.RoutingIndexManager
    routing : ortools.constraint_solver.pywrapcp.RoutingModel
    solution : ortools.constraint_solver.pywrapcp.Assignment

    Returns
    -------
    plan_output : str
        the result of the problem solving
    """
    vehicle_id = 0
    index = routing.Start(vehicle_id)
    plan_output = '\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} -> '.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
    plan_output += '{}\n'.format(manager.IndexToNode(index))

    return plan_output


def best_path(model):
    """
    Final solver of the supermarket problem

    Parameters
    -------
    model : Model (new class)
        the complete model to solve

    Returns
    -------
    final_output : dict
        the result of the problem solving
    """
    # Instantiate the data problem.
    data = create_data_model(model)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'],
                                           data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        the_path = display_solution(
            manager, routing, solution).split('\n')[1]
        the_path = the_path.replace(" ", "").split('->')
        Z = []
        for i in the_path:
            index_i = int(i)
            if index_i == len(model.U):
                Z.append(model.start)
            elif index_i == len(model.U) + 1:
                Z.append(model.end)
            else:
                Z.append(model.U[index_i])
        # -Construction
        final_output = {}
        for i in range(len(Z) - 1):
            begin = Z[i]
            final = Z[i + 1]
            edge_1 = begin + '-' + final
            edge_2 = final + '-' + begin
            edge = ''
            if edge_1 in data['path']:
                edge = edge_1
            else:  # <=> elif edge_2 in data['path']
                edge = edge_2
            for e in data['path'][edge]:
                if not(
                    contains_vertex(
                        e,
                        's')) and not(
                    contains_vertex(
                        e,
                        't')):
                    if e in final_output:
                        final_output[e] += 1
                    else:
                        final_output[e] = 1
        print(type(final_output))
        return final_output
    else:
        return '!!! No solution !!!'


# ------------------------------------------
# -------------Test-------------
def test():
    """ Test the code """
    assert contains_vertex('a-b', 'a')
    assert not contains_vertex('a-b', 'c')
    assert contains_vertex_first('a-b', 'a')
    assert not contains_vertex_first('a-b', 'b')
    assert not contains_vertex_second('a-b', 'a')
    assert contains_vertex_second('a-b', 'b')

    graph = {'A': ['B', 'D'],
         'B': ['A', 'C'],
         'C': ['B', 'D'],
         'D': ['A', 'C']}
    start = 'A'
    end = 'C'
    U = ['D', 'B']

    supermarket = Model(graph, start, end, U)
    solution = best_path(supermarket)
    assert solution == {'B-A': 1, 'C-B': 1, 'D-C': 2}

    print('~ All is ok ~')
