def and_postings(a: list[str], b: list[str]) -> list[str]:
    result: list[str] = []
    left_index = 0
    right_index = 0

    while left_index < len(a) and right_index < len(b):
        if a[left_index] < b[right_index]:
            left_index += 1
        elif a[left_index] > b[right_index]:
            right_index += 1
        else:
            result.append(a[left_index])
            left_index += 1
            right_index += 1

    return result


def or_postings(a: list[str], b: list[str]) -> list[str]:
    result: list[str] = []
    left_index = 0
    right_index = 0

    while left_index < len(a) and right_index < len(b):
        if a[left_index] < b[right_index]:
            result.append(a[left_index])
            left_index += 1
        elif a[left_index] > b[right_index]:
            result.append(b[right_index])
            right_index += 1
        else:
            result.append(a[left_index])
            left_index += 1
            right_index += 1

    result.extend(a[left_index:])
    result.extend(b[right_index:])
    return result


def not_postings(a: list[str], universe: list[str]) -> list[str]:
    result: list[str] = []
    posting_index = 0
    universe_index = 0

    while universe_index < len(universe):
        if posting_index >= len(a) or universe[universe_index] < a[posting_index]:
            result.append(universe[universe_index])
            universe_index += 1
        elif universe[universe_index] == a[posting_index]:
            posting_index += 1
            universe_index += 1
        else:
            posting_index += 1

    return result
