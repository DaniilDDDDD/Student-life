import csv


def parse_txt(filename, oriented=True):
    """
    Parse data from txt file into dict python type.
    JSON serializable.
    """
    data = {}
    with open(filename) as file:

        line = file.readline()
        while line:

            # skip comments
            if line[0] == '#':
                line = file.readline()
                continue

            parent, child = line.split()
            parent = int(parent)
            child = int(child)

            # rows in data file can be duplicated
            if parent in data:
                if child not in data[parent]['linked']:
                    data[parent]['linked'].append(child)
                    data[parent]['degree'] += 1
            else:
                data[parent] = {
                    'linked': [child],
                    'distances': {},
                    'degree': 1,
                    'centrality': 0,
                    'marked': False,
                    'active': True
                }

            if oriented:
                if child not in data:
                    data[child] = {
                        'linked': [],
                        'distances': {},
                        'degree': 0,
                        'centrality': 0,
                        'marked': False,
                        'active': True
                    }

            else:
                if child in data:
                    if parent not in data[child]['linked']:
                        data[child]['linked'].append(parent)
                        data[child]['degree'] += 1

                else:
                    data[child] = {
                        'linked': [parent],
                        'distances': {},
                        'degree': 1,
                        'centrality': 0,
                        'marked': False,
                        'active': True
                    }

            line = file.readline()

    return data


def parse_csv(filename, oriented=True):
    data = {}

    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:

            parent = int(row[0])
            child = int(row[1])

            if parent in data:
                if child not in data[parent]['linked']:
                    data[parent]['linked'].append(child)
                    data[parent]['degree'] += 1
            else:
                data[parent] = {
                    'linked': [child],
                    'distances': {},
                    'degree': 1,
                    'centrality': 0,
                    'marked': False,
                    'active': True
                }

            if oriented:
                if child not in data:
                    data[child] = {
                        'linked': [],
                        'distances': {},
                        'degree': 0,
                        'centrality': 0,
                        'marked': False,
                        'active': True
                    }

            else:
                if child in data:
                    if parent not in data[child]['linked']:
                        data[child]['linked'].append(parent)
                        data[child]['degree'] += 1

                else:
                    data[child] = {
                        'linked': [parent],
                        'distances': {},
                        'degree': 1,
                        'centrality': 0,
                        'marked': False,
                        'active': True
                    }

    return data


def parse(filename, oriented=True):
    if filename.split('.')[-1] == 'txt':
        return parse_txt(filename, oriented)
    elif filename.split('.')[-1] == 'csv':
        return parse_csv(filename, oriented)
