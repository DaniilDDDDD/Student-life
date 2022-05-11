import random
from itertools import product
import matplotlib.pyplot as plt

from parsers import parse
from distance_counters import bfs, shortest_path
from landmarks_selection import select_landmarks


FILENAME = '../datasets/Wiki-Vote.txt'
ORIENTED = True
LANDMARKS_PERCENT = 10
TEST_VERTICES_PERCENT = 10
USE_VERTICES_FROM_ONE_CONNECTIVITY_COMPONENT = False
SAVE_PLOTS = True
SAVE_FOLDER = 'wiki-graph-results/'


data = parse(FILENAME, ORIENTED)
data_items = data.items()
graph_size = len(data_items)
print('Number of vertices in graph: ' + str(graph_size))

select_strategies = ('random', 'degree', 'closeness')
estimation_strategies = ('geometrical_mean', 'middle_point', 'upper', 'lower')

test_vertices_number = int(
    graph_size * (TEST_VERTICES_PERCENT / 100)
) if TEST_VERTICES_PERCENT > 1 else int(
    graph_size * TEST_VERTICES_PERCENT
)

results = {
    'random': {},
    'degree': {},
    'closeness': {}
}

# more then O(n^5)
if USE_VERTICES_FROM_ONE_CONNECTIVITY_COMPONENT:
    vertices = [i[0] for i in data_items]
    prod = list(product(vertices, vertices))
    random.shuffle(prod)
    vertices_source = []
    vertices_stock = []
    while len(vertices_source) < test_vertices_number and prod:
        combination = prod.pop(0)
        source = combination[0]
        stock = combination[1]
        if source != stock and bfs(source, stock, data) != -1:
            vertices_source.append(source)
            vertices_stock.append(stock)
            print('=' * 126)
            print(source)
            print(stock)
            print('Source and stock vertices is on ' + str(len(vertices_source) / test_vertices_number) + ' full.')

else:
    vertices = [i[0] for i in data_items]
    vertices_source = vertices.copy()
    random.shuffle(vertices_source)
    vertices_source = vertices_source[:test_vertices_number]

    vertices_stock = vertices.copy()
    random.shuffle(vertices_stock)
    vertices_stock = vertices_stock[:test_vertices_number]

# count test results for each type of calculation

for select in select_strategies:

    print('=' * 125)
    print(select)

    for percent in range(10, 60, 10):

        results[select][percent] = {
            'geometrical_mean': [],
            'middle_point': [],
            'upper': [],
            'lower': [],
            'selection_operations': 0,
            'selection_time': 0
        }

        # rollback data manually to use source data
        for k, v in data.items():
            v['distances'] = {}
            v['centrality'] = 0

        landmarks, operations_landmarks_selection, time_operations_landmarks_selection = select_landmarks(
            data, percent, ranking=select, monitoring=True, rollback=True)

        print('=' * 125)
        print(landmarks)

        results[select][percent]['selection_operations'] = operations_landmarks_selection
        results[select][percent]['selection_time'] = time_operations_landmarks_selection.total_seconds() * 1000

        for estimation in estimation_strategies:

            print('=' * 125)
            print(estimation)

            for source, stock in zip(vertices_source, vertices_stock):
                print('=' * 125)
                print('Source: ' + str(source))
                print('Stock: ' + str(stock))

                distance_bfs, operations_bfs, time_bfs = bfs(source, stock, data, monitoring=True)

                distance_landmarks, operations_landmarks, time_landmarks = shortest_path(
                    source, stock, landmarks, data, estimation_strategy=estimation, monitoring=True
                )

                print('Distance BFS:' + str(distance_bfs))
                print('Distance Landmarks: ' + str(distance_landmarks))

                accuracy = abs(distance_bfs - distance_landmarks)
                operation_delta = abs(operations_bfs - operations_landmarks)
                time_delta = abs(time_bfs - time_landmarks)

                print('Accuracy: ' + str(accuracy))
                print('Operations delta: ' + str(operation_delta))
                print('Operating time delta: ' + str(time_delta))

                results[select][percent][estimation].append((accuracy, operation_delta, time_delta))

'''
plot_data = {
    'geometrical_mean': {
        'random': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'degree': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'closeness': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        }
    },
    'middle_point': {
        'random': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'degree': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'closeness': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        }
    },
    'upper': {
        'random': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'degree': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'closeness': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        }
    },
    'lower': {
        'random': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'degree': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        },
        'closeness': {
            'accuracy': {},
            'time_delta': {},
            'operations_delta': {}
        }
    }
}

# key - select strategy
for strategy, strategy_value in results.items():
    # percent - percent of landmarks
    for percent, percent_value in strategy_value.items():
        # rstimation - estimation stratery
        for estimation, metrics in percent_value.items():

            try:
                sum_accuracy = 0
                sum_operations_delta = 0
                sum_time_delta = 0
                for metric in metrics:
                    sum_accuracy += metric[0]
                    sum_operations_delta += metric[1]
                    sum_time_delta += metric[2].total_seconds() * 1000  # in milliseconds

                plot_data[estimation][strategy]['accuracy'][percent] = sum_accuracy / len(metrics)
                plot_data[estimation][strategy]['time_delta'][percent] = sum_time_delta / len(metrics)
                plot_data[estimation][strategy]['operations_delta'][percent] = sum_operations_delta / len(metrics)

            except TypeError:
                continue

# print(plot_data)


labels = [key for key, value in plot_data['geometrical_mean']['random']['accuracy'].items()]
colors = {'random landmarks': 'blue', 'top degree landmarks': 'green', 'less closeness landmarks': 'red'}

# Landmarks selection operations number

plt.figure(figsize=(16, 9))

y1 = [value['selection_operations'] for percent, value in results['random'].items()]
y2 = [value['selection_operations'] for percent, value in results['degree'].items()]
y3 = [value['selection_operations'] for percent, value in results['closeness'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operations number')
plt.title('Landmarks selection operations number with differend strategies', fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)
if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'selection_operations.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Landmarks selection operating time

plt.figure(figsize=(16, 9))

y1 = [value['selection_time'] for percent, value in results['random'].items()]
y2 = [value['selection_time'] for percent, value in results['degree'].items()]
y3 = [value['selection_time'] for percent, value in results['closeness'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operating time')
plt.title('Landmarks selection operating time with differend strategies', fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'selection_time.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Accuracy error with geometrical mean estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['geometrical_mean']['random']['accuracy'].items()]
y2 = [value for key, value in plot_data['geometrical_mean']['degree']['accuracy'].items()]
y3 = [value for key, value in plot_data['geometrical_mean']['closeness']['accuracy'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Accuracy error (in distance)')
plt.title('Accuracy error with different strategies of landmarks selection using geometrical mean distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'accuracy_geometrical_mean.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Accuracy error with middle point estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['middle_point']['random']['accuracy'].items()]
y2 = [value for key, value in plot_data['middle_point']['degree']['accuracy'].items()]
y3 = [value for key, value in plot_data['middle_point']['closeness']['accuracy'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Accuracy error (in distance)')
plt.title('Accuracy error with different strategies of landmarks selection using middle point distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'accuracy_middle_point.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Accuracy error with upper estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['upper']['random']['accuracy'].items()]
y2 = [value for key, value in plot_data['upper']['degree']['accuracy'].items()]
y3 = [value for key, value in plot_data['upper']['closeness']['accuracy'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Accurasy error (in distance)')
plt.title('Accuracy error with different strategies of landmarks selection using upper distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'accuracy_upper.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Accuracy error with lower estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['lower']['random']['accuracy'].items()]
y2 = [value for key, value in plot_data['lower']['degree']['accuracy'].items()]
y3 = [value for key, value in plot_data['lower']['closeness']['accuracy'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Accuracy error (in distance)')
plt.title('Accuracy error with different strategies of landmarks selection using lower distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'accuracy_lower.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations delta with geometrical mean estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['geometrical_mean']['random']['operations_delta'].items()]
y2 = [value for key, value in plot_data['geometrical_mean']['degree']['operations_delta'].items()]
y3 = [value for key, value in plot_data['geometrical_mean']['closeness']['operations_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operations delta (operations number)')
plt.title(
    'Operations delta with differend strategies of landmarks selection using geometrical mean distance estimation',
    fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'operations_delta_geometrical_mean.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations delta with middle point estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['middle_point']['random']['operations_delta'].items()]
y2 = [value for key, value in plot_data['middle_point']['degree']['operations_delta'].items()]
y3 = [value for key, value in plot_data['middle_point']['closeness']['operations_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operations delta (operations number)')
plt.title('Operations delta with different strategies of landmarks selection using middle point distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'operations_delta_middle_point.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations delta with upper estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['upper']['random']['operations_delta'].items()]
y2 = [value for key, value in plot_data['upper']['degree']['operations_delta'].items()]
y3 = [value for key, value in plot_data['upper']['closeness']['operations_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operations delta (operations number)')
plt.title('Operations delta with different strategies of landmarks selection using upper distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'operations_delta_upper.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations delta with lower estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['lower']['random']['operations_delta'].items()]
y2 = [value for key, value in plot_data['lower']['degree']['operations_delta'].items()]
y3 = [value for key, value in plot_data['lower']['closeness']['operations_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operations delta (operations number)')
plt.title('Operations delta with different strategies of landmarks selection using lower distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'operations_delta_lower.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations time delta with geometrical mean estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['geometrical_mean']['random']['time_delta'].items()]
y2 = [value for key, value in plot_data['geometrical_mean']['degree']['time_delta'].items()]
y3 = [value for key, value in plot_data['geometrical_mean']['closeness']['time_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operating time delta (milliseconds)')
plt.title('Operating time delta with different strategies of landmarks selection using lower distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'time_delta_geometrical_mean.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations time delta with middle point estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['middle_point']['random']['time_delta'].items()]
y2 = [value for key, value in plot_data['middle_point']['degree']['time_delta'].items()]
y3 = [value for key, value in plot_data['middle_point']['closeness']['time_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operating time delta (milliseconds)')
plt.title('Operating time delta with different strategies of landmarks selection using lower distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'time_delta_middle_point.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations time delta with upper estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['upper']['random']['time_delta'].items()]
y2 = [value for key, value in plot_data['upper']['degree']['time_delta'].items()]
y3 = [value for key, value in plot_data['upper']['closeness']['time_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operating time delta (milliseconds)')
plt.title('Operating time delta with different strategies of landmarks selection using lower distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'time_delta_upper.jpg', dpi=100, bbox_inches='tight')
plt.show()

# Operations time delta with lower estimation

plt.figure(figsize=(16, 9))

y1 = [value for key, value in plot_data['lower']['random']['time_delta'].items()]
y2 = [value for key, value in plot_data['lower']['degree']['time_delta'].items()]
y3 = [value for key, value in plot_data['lower']['closeness']['time_delta'].items()]

plt.plot(labels, y1, color='red')
plt.plot(labels, y2, color='green')
plt.plot(labels, y3, color='blue')

plt.xlabel('Percent of landmarks')
plt.ylabel('Operating time delta (milliseconds)')
plt.title('Operating time delta with different strategies of landmarks selection using lower distance estimation',
          fontsize=30)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, color=colors[color[0]]) for color in colors.items()],
    list(colors.keys())
)

if SAVE_PLOTS:
    plt.savefig(SAVE_FOLDER + 'time_delta_lower.jpg', dpi=100, bbox_inches='tight')
plt.show()
'''