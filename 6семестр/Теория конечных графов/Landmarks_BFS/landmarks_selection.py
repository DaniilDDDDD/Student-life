import random
from math import log
import datetime as dt
from distance_counters import count_distance


def select_landmarks(
        data: dict,
        oriented: bool = False,
        number_of_landmarks: float = 0.1,
        ranking: str = 'degree',
        h: int = 1,
        rollback: bool = False,
        monitoring: bool = False
):
    """
    Select landmarks using constrained strategy with provided h and ranking parameters.
    Set h to -1 to get top 'number_of_landmarks' ranked vertices.
    Possible ranking: degree, random, closeness.
    Set rollback to True to roll back all vertices 'active' to True
    """
    operations_counter = 0
    timestamp_before_algorithm = dt.datetime.now()

    data_items = data.items()
    graph_size = len(data_items)
    #     print('=' * 126)
    #     print('Graph size: ' + str(graph_size))

    number_of_landmarks = int(
        graph_size * (number_of_landmarks / 100)
    ) if number_of_landmarks >= 1 else int(
        graph_size * number_of_landmarks
    )

    #     print('=' * 126)
    #     print('Target number of landmarks: ' + str(number_of_landmarks))

    landmarks = []

    operations_counter += graph_size + 5

    if ranking == 'random':

        vertices = [i[0] for i in data_items]
        if h == -1:

            random.shuffle(vertices)
            random_vertices = vertices[:number_of_landmarks]

            if not oriented:

                for v in random_vertices:
                    operations, __ = count_distance(vertice=v, data=data, h=-1, full=True, monitoring=True)
                    operations_counter += operations + 1

                operations_counter += 1 + number_of_landmarks

            else:
                for v in random_vertices:


            if monitoring:
                return random_vertices, operations_counter, dt.datetime.now() - timestamp_before_algorithm
            return random_vertices

        operations_counter += len(data_items) + number_of_landmarks

        random.shuffle(vertices)

        operations_counter += len(vertices)

        while len(landmarks) < number_of_landmarks and vertices:

            v = vertices.pop(0)

            # check if vertice 'v' is less than 'h' away from some previously selected landmark
            if not data[v]['active']:
                continue

            landmarks.append(v)
            _, operations, __ = count_distance(vertice=v, data=data, h=h, full=True, monitoring=True)

            operations_counter += operations + 3


    elif ranking == 'degree':

        data_sorted = sorted(data_items, key=lambda x: x[1]['degree'], reverse=True)

        operations_counter += graph_size * log(graph_size)

        if h == -1:

            max_degree_vertices = [i[0] for i in data_sorted[:number_of_landmarks]]

            for v in max_degree_vertices:
                operations, __ = count_distance(vertice=v, data=data, h=-1, full=True, monitoring=True)
                operations_counter += operations + 1

            operations_counter += 2 * number_of_landmarks + 1

            if monitoring:
                return max_degree_vertices, operations_counter, dt.datetime.now() - timestamp_before_algorithm
            return max_degree_vertices

        while len(landmarks) < number_of_landmarks and data_sorted:
            v = data_sorted.pop(0)[0]

            # check if vertice 'v' is less than 'h' away from some previously selected landmark
            if not data[v]['active']:
                continue

            landmarks.append(v)
            _, operations, __ = count_distance(vertice=v, data=data, h=h, full=True, monitoring=True)

            operations_counter += operations + 3


    elif ranking == 'closeness':

        shuffled_vertices = [i[0] for i in data_items]
        random.shuffle(shuffled_vertices)

        number_of_seeds = int(
            number_of_landmarks + (graph_size - number_of_landmarks) / 2
        ) if number_of_landmarks >= (graph_size / 2) else number_of_landmarks * 2
        #         print('=' * 126)
        #         print('Number of seeds: ' + str(number_of_seeds))

        operations_counter += 2 * graph_size + 6

        if h == -1:

            seeds = []
            while len(seeds) < number_of_seeds and shuffled_vertices:
                seed = shuffled_vertices.pop(0)
                if data[seed]['linked']:
                    operations, __ = count_distance(vertice=seed, data=data, h=-1, full=True, monitoring=True)
                    seeds.append(seed)

                    operations_counter += operations + 1
                operations_counter += 1

            # select top 'number_of_landmarks' vertices with lowest closeness centrality
            min_centrality_vertices = [
                i[0] for i in sorted(
                    [(key, value['centrality']) for key, value in data.items() if key in seeds],
                    key=lambda x: x[1]
                )[:number_of_landmarks]
            ]

            operations_counter += len(seeds) + log(len(seeds)) * len(seeds)

            if monitoring:
                return min_centrality_vertices, operations_counter, dt.datetime.now() - timestamp_before_algorithm
            return min_centrality_vertices

        seeds = []
        while len(seeds) < number_of_seeds and shuffled_vertices:
            seed = shuffled_vertices.pop(0)
            if data[seed]['linked']:
                _, operations, __ = count_distance(vertice=seed, data=data, h=h, full=True, monitoring=True)
                seeds.append(seed)

                operations_counter += operations + 1
            operations_counter += 1

        # sort seeds by closeness centrality
        seeds_sorted = [
            i[0] for i in sorted(
                [(key, value['centrality']) for key, value in data.items() if key in seeds],
                key=lambda x: x[1]
            )
        ]

        #         for s in seeds_sorted:
        #             print(str(s))
        #             print(data[s]['centrality'])
        #             print(data[s]['linked'])
        #             print()

        #         print('=' * 100)

        landmarks.append(seeds_sorted.pop(0))
        #         print(str(landmarks[0]) + ': ' + str(data[landmarks[0]]['active']))

        operations_counter += graph_size + len(seeds) * log(len(seeds))

        while len(landmarks) < number_of_landmarks:

            if not seeds_sorted:
                #                 print('=' * 126)
                #                 print('For unsortet vertices: ')
                #                 print("Number of landmarks after iteration by sorted seeds: " + str(len(landmarks)))
                #                 print('=' * 126)

                # iteration by remaining shuffled vertices
                for seed in shuffled_vertices:
                    if len(landmarks) >= number_of_landmarks:
                        break
                    if data[seed]['active']:
                        _, operations, __ = count_distance(vertice=seed, data=data, h=h, full=True, monitoring=True)
                        landmarks.append(seed)
                        operations_counter += operations + 1

                    operations_counter += 2
                break

            landmarks.append(seeds_sorted.pop(0))

            operations_counter += 3

    #     print('Target number of landmarks: ' + str(number_of_landmarks))
    #     print('Number of calculated landmarks: ' + str(len(landmarks)))

    # data rollback
    if rollback:
        for key, value in data.items():
            value['active'] = True

    if monitoring:
        return landmarks, operations_counter, dt.datetime.now() - timestamp_before_algorithm
    return landmarks
