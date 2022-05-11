# from distance_counters import count_distance
from landmarks_selection import select_landmarks
from parsers import parse


def main():
    data = parse('../datasets/Wiki-Vote.txt', True)

    landmarks = select_landmarks(
        data,
        10,
        ranking='random',
        monitoring=False,
        rollback=False
    )

    print(landmarks)

    for key, value in sorted(data.items(), key=lambda x: x[0]):
        print(key)
        print(value)
        print()

if __name__ == '__main__':
    main()
