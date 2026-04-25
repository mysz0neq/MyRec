"""Cleans data"""

from collections import defaultdict
from numpy import mean


def iteration(data: list[tuple[str | int, str | int, float]],
              min_u: int,
              min_f: int) -> list[tuple[str | int, str | int, float]]:
    """Single iteration of K-core filtering algorithm"""
    counter_u, counter_f = counters(data)

    new_data = []
    for u, f, r in data:
        if counter_u[u] >= min_u and counter_f[f] >= min_f:
            new_data.append((u, f, r))
    return new_data


def counters(data: list[tuple[str | int, str | int, float]]) -> tuple[
    defaultdict[str | int, int], defaultdict[str | int, int]]:
    """Returns counter dictionaries with number of records for each user and movie. The order is (counters_u,counters_f)"""

    counter_u = defaultdict(int)
    counter_f = defaultdict(int)
    for u, f, _ in data:
        counter_u[u] += 1
        counter_f[f] += 1
    return counter_u, counter_f


def means(data: list[tuple[str | int, str | int, float]]) -> tuple[dict[str | int, float], dict[str | int, float]]:
    """Returns dictionaries that hold average rating for each user and movie. The order is (means_u,means_f)"""
    ratings_u = defaultdict(list)
    ratings_f = defaultdict(list)
    for u, f, r in data:
        ratings_u[u].append(r)
        ratings_f[f].append(r)
    means_u = {u: float(mean(ratings_u[u])) for u in ratings_u}
    means_f = {f: float(mean(ratings_f[f])) for f in ratings_f}
    return means_u, means_f


def kcore_filter(data: list[tuple[str | int, str | int, float]],
                 min_u: int,
                 min_f: int) -> list[tuple[str | int, str | int, float]]:
    """Implements K-core filtering algorithm to clean
    data records from users with less than min_u
    ratings and movies less than min_f ratings"""

    previous = 0
    current = len(data)

    while previous != current:
        data = iteration(data, min_u, min_f)
        previous = current
        current = len(data)
    return data
