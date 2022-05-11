from collections import deque
import datetime as dt


# Distance counting method with modifications
def count_distance(vertice, data, h=-1, full=False, rollback=True, monitoring=False):
    """
    Counts distances form given vertice to all other in connectivity component that vertice belongs to.
    Also, of h parameter is provided, this method finds list of vertices which are h or less away from provided vertice.
    (As only distance from provided vertive becomes more then h method stops.)
    Based on BFS.
    vertice: index of source vertice
    data: dict with information about graph
    h: distance to closest vertices
    fill: complete BFS in spite of current distance > h
    """

    operations_counter = 0
    timestamp_before_algorithm = dt.datetime.now()

    current_distance = 0
    centrality = 0
    vertices_number = 1
    nearest_vertices = []
    d0 = deque()
    d1 = deque()

    source_active = data[vertice]['active']

    d0.append(vertice)
    data[vertice]['marked'] = True

    operations_counter += 9

    while True:

        if (not d0 and not d1) or (h != -1 and current_distance > h and not full):
            operations_counter += 9
            break

        if current_distance % 2 == 0:

            v = d0.popleft()

            for i in data[v]['linked']:
                if not data[i]['marked']:
                    d1.append(i)
                    data[i]['marked'] = True

                    operations_counter += 2
                operations_counter += 2

            data[v]['distances'][vertice] = current_distance
            vertices_number += 1
            centrality += current_distance

            # set active to false if it vertice 'v' is too close to source vertice 'vertice'
            if h != -1 and current_distance <= h:
                data[v]['active'] = False
                nearest_vertices.append(v)

                operations_counter += 2

            # go to the next level of distance from the source vertice 'vertice'
            if not d0:
                current_distance += 1

                operations_counter += 1

            operations_counter += 5  # number of "if"'s

        else:

            v = d1.popleft()

            for i in data[v]['linked']:
                if not data[i]['marked']:
                    d0.append(i)
                    data[i]['marked'] = True

                    operations_counter += 2
                operations_counter += 2

            data[v]['distances'][vertice] = current_distance
            vertices_number += 1
            centrality += current_distance

            # set active to false if it vertice 'v' is too close to source vertice 'vertice'
            if h != -1 and current_distance <= h:
                data[v]['active'] = False
                nearest_vertices.append(v)

                operations_counter += 3

            # go to the next level of distance from the source vertice 'vertice'
            if not d1:
                current_distance += 1

                operations_counter += 1

            operations_counter += 5  # number of "if"'s added

    # rollback data
    if rollback:
        for key, value in data.items():
            value['marked'] = False

    # set initial status
    data[vertice]['active'] = source_active

    operations_counter = + 1

    if full:
        data[vertice]['centrality'] = centrality / vertices_number
        operations_counter += 4

        if h == -1:
            if monitoring:
                return operations_counter, dt.datetime.now() - timestamp_before_algorithm
        else:
            if monitoring:
                return nearest_vertices, operations_counter, dt.datetime.now() - timestamp_before_algorithm
            return nearest_vertices
    else:
        if h == -1:

            data[vertice]['centrality'] = centrality / vertices_number
            operations_counter += 5

            if monitoring:
                return operations_counter, dt.datetime.now() - timestamp_before_algorithm
        else:
            if monitoring:
                return nearest_vertices, operations_counter, dt.datetime.now() - timestamp_before_algorithm
            return nearest_vertices


def count_distance_to_landmarks(vertice, landmarks, data, rollback=True, monitoring=False):
    """
    Count distances to list of landmarks and save them to vertice data.
    vertice: source vertice index in graph data dict
    landmarks: list of landmarks
    data: graph data dict
    """
    operations_counter = 0
    timestamp_before_algorithm = dt.datetime.now()

    if not all(landmark in data for landmark in landmarks) or vertice not in data:
        print('One or more of provided vertices are not found!')
        operations_counter += 2 * len(data.items())
        if monitoring:
            return -1, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return -1

    current_distance = 0
    flag_found = False
    d0 = deque()
    d1 = deque()

    d0.append(vertice)
    data[vertice]['marked'] = True

    operations_counter += 7

    while True:

        if not d0 and not d1:
            current_distance = -1

            operations_counter += 3
            break

        if current_distance % 2 == 0:

            v = d0.popleft()

            if v in landmarks:
                data[v]['distances'][vertice] = current_distance
                operations_counter += 1
                break

            for i in data[v]['linked']:
                if not data[i]['marked']:
                    d1.append(i)
                    data[i]['marked'] = True

                    operations_counter += 2
                operations_counter += 2

            if not d0:
                current_distance += 1

                operations_counter += 1

            operations_counter += 3

        else:

            v = d1.popleft()

            if v in landmarks:
                data[v]['distances'][vertice] = current_distance
                operations_counter += 1
                break

            for i in data[v]['linked']:
                if not data[i]['marked']:
                    d0.append(i)
                    data[i]['marked'] = True

                    operations_counter += 2
                operations_counter += 2

            if not d1:
                current_distance += 1

                operations_counter += 1

            operations_counter += 3

    if rollback:
        for key, value in data.items():
            value['marked'] = False

    if flag_found:
        if monitoring:
            return current_distance, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return current_distance
    else:
        if monitoring:
            return -1, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return -1


def bfs(source, stock, data, rollback=True, monitoring=False):
    """
    Count distance from source to stock without using landmarks.
    Pure BFS.
    """
    operations_counter = 0
    timestamp_before_algorithm = dt.datetime.now()

    if not all(key in data for key in (source, stock)):
        print('Key not found')
        operations_counter += 2 * len(data.items())
        if monitoring:
            return -1, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return -1

    current_distance = 0
    flag_found = False
    d0 = deque()
    d1 = deque()

    d0.append(source)
    data[source]['marked'] = True

    operations_counter += 7

    while True:

        if not d0 and not d1:
            current_distance = -1

            operations_counter += 3
            break

        if current_distance % 2 == 0:

            v = d0.popleft()

            if v == stock:
                flag_found = True

                operations_counter += 1
                break

            for i in data[v]['linked']:
                if not data[i]['marked']:
                    d1.append(i)
                    data[i]['marked'] = True

                    operations_counter += 2
                operations_counter += 2

            if not d0:
                current_distance += 1

                operations_counter += 1

            operations_counter += 3

        else:

            v = d1.popleft()

            if v == stock:
                flag_found = True
                break

            for i in data[v]['linked']:
                if not data[i]['marked']:
                    d0.append(i)
                    data[i]['marked'] = True

                    operations_counter += 2
                operations_counter += 2

            if not d1:
                current_distance += 1

                operations_counter += 1

            operations_counter += 3

    if rollback:
        for key, value in data.items():
            value['marked'] = False

    if flag_found:
        if monitoring:
            return current_distance, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return current_distance
    else:
        if monitoring:
            return -1, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return -1


# Shortest path estimating via landmarks method
def shortest_path(
        source,
        stock,
        landmarks,
        data,
        estimation_strategy='geometrical_mean',
        monitoring=False
):
    """
    Counts distance from source to stock using landmarks.
    For distance estimation geometric mean is used.
    There are 4 estimation strategies: geometrical mean, upper, lower and middle point. Geometrical mean is default.
    """

    operations_counter = 0
    timestamp_before_algorithm = dt.datetime.now()

    # source or stock not in graph
    try:
        source_distances = data[source]['distances']
        stock_distances = data[stock]['distances']
    except KeyError:
        if monitoring:
            return -1, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return -1

    graph_size = len(data.items())

    L = -1
    U = 3 * graph_size

    operations_counter += 5

    #     for landmark in landmarks:
    for key, to_source in source_distances.items():

        # use distances only to landmarks

        #         to_source = source_distances.get(landmark, -1)
        #         to_stock = stock_distances.get(landmark, -1)
        #         if to_source == -1 or to_stock == -1:
        #             continue
        to_stock = stock_distances.get(key, -1)
        if to_stock == -1:
            continue

        l = abs(to_source - to_stock)
        u = to_source + to_stock

        if l > L:
            L = l
            operations_counter += 1
        if u < U:
            U = u
            operations_counter += 1

        operations_counter += 9 + len(landmarks)

    if L == -1 and U == 3 * graph_size:
        # this mean that source and stock are in different connectivity components

        operations_counter += 3

        if monitoring:
            return -1, operations_counter, dt.datetime.now() - timestamp_before_algorithm
        return -1

    # choose estimating strategy
    if estimation_strategy == 'geometrical_mean':
        result = (L * U) ** 0.5
        operations_counter += 3
    elif estimation_strategy == 'middle_point':
        result = (L + U) / 2
        operations_counter += 3
    elif estimation_strategy == 'upper':
        result = U
        operations_counter += 2
    elif estimation_strategy == 'lower':
        result = L
        operations_counter += 2
    else:
        result = (L * U) ** 0.5
        operations_counter += 3

    if monitoring:
        return result, operations_counter, dt.datetime.now() - timestamp_before_algorithm
    return result
