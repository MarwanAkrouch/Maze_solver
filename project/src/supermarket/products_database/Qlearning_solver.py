import numpy as np

## train for solving sorthest path

def sample_next_action(R, Q, state, epsilon):
    """
    This function chooses at random which action to be performed within the range
    of all the available actions.

    Parameters
    --------
    R: reward matrix
    Q: state-action matrix
    state: current state
    epsilon: Exploration parameter

    returns
    --------
    an available action
    """
    current_state_row = R[state,]
    available_act = np.where(current_state_row >= -1)[1]
    if np.random.random()<epsilon:
        next_action = int(np.random.choice(available_act,1))
    else:
        max_index = np.where(Q[state, ] == np.max(Q[state, ]))[1]

        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size = 1))
        else:
            max_index = int(max_index)
        next_action = max_index

    return next_action


def update(R, Q, current_state, action, gamma, alpha):
    """
    This function updates the Q matrix according to the path selected and the
    Q learning algorithm

    Parameters
    --------
    R: reward matrix
    Q: state-action matrix
    current_state: current state
    action: action to take
    gamma: discount factor
    alpha: learning rate

    """
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    # Q learning formula
    Q[current_state, action] += alpha*( R[current_state, action] + gamma * max_value - Q[current_state, action])

print("training the model...")

def pairToPair_shortestPaths(width, height, graph, entry, exit, target_products):
    '''
    function giving shortest paths between each two target products and considers also entry and exit

    Parameters
    ---------
    width: int defining width of the maze
    height: int defining height of the maze
    graph: the graph defined by the maze
    entry, exit:  tuples (i,j) = where the entry (resp. exit) are located
    target_products: tuples (i,j) = where target products are located on the maze

    returns
    ----------
    a dictionary such that for (i1,j1) ,(i2,j2) in target_products:
    dict[(i1,j1) ,(i2,j2)] is a list of tuples describing the shortest path to follow
    to get from (i1,j1) to (i2,j2)
    If (i1,j1) ,(i2,j2) is in the dictionary keys then (i2,j2), (i1,j1) is not
    since we don't need to calculate the path gain
    '''

    products = [entry] + list(set(target_products)) + [exit]
    pair_products = [(x,y) for x in products for y in products if x!=y]
    for elem1 in pair_products:
        for elem2 in pair_products:
            if set(elem1) == set(elem2) and elem1 != elem2:
                pair_products.remove(elem2)

    pair_to_pair_shortestPaths = {}

    for elem in pair_products:
        start, dest = elem
        # reward matrix
        R = np.matrix(np.full(((width)*(height), (width)*(height)), -np.Inf))
        for i in range(width):
            for j in range(height):
                if j>0:
                    R[i + width*j, i + width* (j-1)] = (5000 if (i, j-1) == dest else -1) if (i, j-1) in graph[i,j] else -np.Inf
                if j<height-1:
                    R[i + width*j, i + width* (j+1)] = (5000 if (i, j+1) == dest else -1) if (i, j+1) in graph[i,j] else -np.Inf
                if i>0:
                    R[i + width*j, i-1 + width* j]   = (5000 if (i-1, j) == dest else -1) if (i-1, j) in graph[i,j] else -np.Inf
                if i<width-1:
                    R[i + width*j, i+1 + width* j]   = (5000 if (i+1, j) == dest  else -1) if (i+1, j) in graph[i,j] else -np.Inf

        Q = np.matrix(np.zeros([(width)*(height),(width)*(height)]))

        # Gamma (discount factor).
        gamma = 0.9

        #learning rate
        alpha = 0.2

        #training steps:

        # Initial state. (Usually to be chosen at random)
        initial_state = start[0]+width*start[1]
        # Sample next action to be performed
        action = sample_next_action(R, Q, initial_state, 1.)
        # Update Q matrix
        update(R, Q, initial_state, action, gamma, alpha)

        #-------------------------------------------------------------------------------
        # Training

        # Train over a number of iterations according to the size of the maze. (Re-iterate the process above).
        n_train = int((width+height)/2)*1000
        for i in range(n_train):
            current_state = np.random.randint(0, int(Q.shape[0]))
            action = sample_next_action(R, Q, current_state, 1. - i/n_train)
            update(R, Q, current_state, action, gamma, alpha)

        #-------------------------------------------------------------------------------
        # Now the agent will start from "start" and follow a greedy path
        # maximising Q to get to "dest"

        current_state = start[0] + width*start[1]
        pair_to_pair_shortestPaths[start, dest] = [start]

        k=0
        while current_state != dest[0] + width* dest[1] and k<50:

            next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

            if next_step_index.shape[0] > 1:
                next_step_index = int(np.random.choice(next_step_index, size = 1))
            else:
                next_step_index = int(next_step_index)

            j =  int(next_step_index/width)
            i = next_step_index - width*j
            pair_to_pair_shortestPaths[start, dest].append((i, j))

            current_state = next_step_index
            k += 1
    return pair_to_pair_shortestPaths

## ----------------------------------------------------------------------------------------
## Train for solving the TSP problem

def update_q(q, dist_mat, state, action, alpha=0.4, gamma=0.7):
    """ update the Q matrix

    Parameters
    ----------
    q: state-action matrix
    dist_mat: distance matrix to get reward from
    action: action
    alpha: learning rate
    gamma: discount factor

    """
    immed_reward = 1./ (dist_mat[state,action])**3  # Set immediate reward as inverse of distance
    delayed_reward = np.max(q[action])
    q[state,action] += alpha * (immed_reward + gamma * delayed_reward - q[state,action])

def dist_mat(products, pair_to_pair_shortestPaths, width, height):
    ''' matrix of distances between target products '''
    n_dest = len(products)
    dist_mat = np.zeros([n_dest,n_dest])
    for i,elem1 in enumerate(products):
        for j,elem2 in enumerate(products):
            if elem1 != elem2:
                if (elem1, elem2) in pair_to_pair_shortestPaths:
                    d = len(pair_to_pair_shortestPaths[(elem1, elem2)]) - 1
                else:
                    d = len(pair_to_pair_shortestPaths[(elem2, elem1)]) - 1
                dist_mat[i,j] = d
            else:
                dist_mat[i,j] = np.Inf
    for i,elem1 in enumerate(products):
        if elem1 != exit:
            dist_mat[i, int(len(products)-1)] += (width*height)**2
            dist_mat[int(len(products)-1), i] += (width*height)**2
    return dist_mat

def TSP_solver(n_dest, dist_mat):  # n_dest = nombre de destination/ nombre de produits cibles
    """
    solve the TSP given a distance matrix

    Parameters
    --------
    dist_mat: an array such that d[i,j] is the distance between i and j
    n_dest: number of destinations/ number of target products
            entry and exit are considered target product
    returns
    --------
    a trajectory to follow, for example: [1, 0, 3, 4 ,2]
    """
    q = np.zeros([n_dest,n_dest])
    epsilon = 1. # Exploration parameter
    n_train = n_dest*5000 # Number of training iterations
    for i in range(n_train):
        state = np.random.randint(0,n_dest-1) # initial state is random
        traj = [state]
        possible_actions = [ dest for dest in range(n_dest) if dest not in traj and dest!=0]

        while possible_actions: # until all destinations are visited
            #decide next action:
            if np.random.random() < epsilon:  # explore random actions
                    action = np.random.choice(possible_actions)
            else:  # exploit known data from Q matrix
                    next_step_index = np.argmax(q[state, possible_actions])
                    action = possible_actions[next_step_index]
            # update Q: core of the training phase
            update_q(q, dist_mat, state, action)
            traj.append(action)
            state = traj[-1]
            possible_actions = [ dest for dest in range(n_dest) if dest not in traj and dest!=0]
            epsilon = max(1. - 5*i * 1/n_train, 0)

    traj = [0]
    state = 0
    distance_travel = 0.
    possible_actions = [ dest for dest in range(n_dest) if dest not in traj ]
    while possible_actions: # until all destinations are visited
        next_step_index = np.argmax(q[state, possible_actions])
        action = possible_actions[next_step_index]
        distance_travel += dist_mat[state, action]
        traj.append(action)
        state = traj[-1]
        possible_actions = [ dest for dest in range(n_dest) if dest not in traj ]

    return traj

## Combining the two steps above to solve the problem

def QLearning_solver(width, height, graph, entry, exit, target_products):
    """
    Solve the maze problem using Qlearning
    Parameters
    --------
    width, height: int, width and height of the maze
    graph: the graph defined by the maze
    entry, exit:  tuple (i,j) where the entry (resp. exit) are located
    target_products: tuples (i,j) where target products are located on the maze

    returns
    --------
    a path which is a list of tuples (i,j) representing the coordinates of cells
    to follow one bye one
    """
    products = [entry] + list(set(target_products)) + [exit]
    n_dest = len(products)
    pair_to_pair_shortestPaths = pairToPair_shortestPaths(width, height, graph, entry, exit, target_products)

    distmat = dist_mat(products, pair_to_pair_shortestPaths, width, height)

    traj = TSP_solver(n_dest, distmat)

    #Print Results:
    print('Products to pick in order:')
    print(' -> '.join([str(products[b]) for b in traj]))
    #print(f'Distance Travelled: {distance_travel-width*height}')

    # Print selected sequence of steps
    path = []
    path_to_follow = []

    for i in range(len(traj)-1):

        product_pair = (products[traj[i]], products[traj[i+1]])
        inverse_product_pair = (products[traj[i+1]], products[traj[i]])
        if product_pair in pair_to_pair_shortestPaths:
            path_between_product_pair = pair_to_pair_shortestPaths[product_pair]
            path_to_follow += path_between_product_pair
            for k in range(len(path_between_product_pair) - 1):
                path.append((path_between_product_pair[k], path_between_product_pair[k+1]))

        elif inverse_product_pair in pair_to_pair_shortestPaths:
            path_between_product_pair = list(reversed(pair_to_pair_shortestPaths[inverse_product_pair]))
            path_to_follow += path_between_product_pair
            for k in range(len(path_between_product_pair) - 1):
                path.append((path_between_product_pair[k], path_between_product_pair[k+1]))

    index_to_del = []
    for i in range(len(path_to_follow)-1):
        if path_to_follow[i] == path_to_follow[i+1]:
            index_to_del.append(i)
    for index in index_to_del:
        if index<len(path_to_follow):
            del path_to_follow[index]

    print('trajectory to follow:')
    print(' -> '.join([str(b) for b in path_to_follow]))

    return path