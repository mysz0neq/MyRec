"""Gets data ready for analysis and Machine Learning"""

import warnings
from copy import deepcopy
import random

def users_movies_sets(data: list[tuple[str | int, str | int, float]]) -> tuple[set, set]:
    """Returns sets of users and movies in the data."""
    users = set()
    movies = set()
    for u, f, _ in data:
        users.add(u)
        movies.add(f)
    return users, movies

def tokenizer(data: list[tuple[str | int, str | int, float]],
              base_users: set = None,
              base_movies: set = None,
              base_u2i: dict = None,
              base_m2i: dict = None) -> tuple[list[tuple[int, int, float]], dict, dict, dict, dict, set[str], set[str]]:
    """Changes users and movies names to IDs and returns the dictionaries.
    It can also perform a second time tokenization on new data if you specify sets and dictionaries returned by the former tokenization."""
    print("Starting tokenizer...\n")
    current_users, current_movies = users_movies_sets(data)
    print("\nUSERS:")
    if not base_users and base_u2i:
        warnings.warn(
            "Users base dictionary specified without users set given!\nBase users set set to base dictionary keys.")
        base_users = set([u for u in base_u2i.keys()])

    elif base_users and not base_u2i:
        warnings.warn(
            "Base users set specified without users dictionary given!\nBase dictionary created from base users set.")
        base_u2i = {u: i for i, u in enumerate(sorted(list(base_users)))}

    if not base_users and not base_u2i:  # kiedy puszczamy po raz pierwszy - ma wypluć zbiór userów i u2i
        print("Operating in first time run mode.\nCreated users set and dictionary.")
        base_users = current_users
        new_users = set()
        base_u2i = {u: i for i, u in enumerate(sorted(list(base_users)))}

    elif base_users and base_u2i:  # fine tune - ma wypluc zbior userów i u2i
        print("Operating in fine tuning mode.\nCreated new_users set and updated base dictionary.")
        new_users = current_users - base_users
        base_u2i.update({u: i for i, u in enumerate(sorted(list(new_users)), start=len(base_u2i))})
    else:
        raise RuntimeError("Something failed while creating users dictionaries and sets.")

    print("\nFILMS:")
    if not base_movies and base_m2i:
        warnings.warn(
            "Films base dictionary specified without filmss set given!\nBase filmss set set to base dictionary keys.")
        base_movies = set([u for u in base_m2i.keys()])

    elif base_movies and not base_m2i:
        warnings.warn(
            "Base films set specified without films dictionary given!\nBase dictionary created from base films set.")
        base_m2i = {f: i for i, f in enumerate(sorted(list(base_movies)))}

    if not base_movies and not base_m2i:  # kiedy puszczamy po raz pierwszy - ma wypluć zbiór userów i u2i
        print("Operating in first time run mode.\nCreated films set and dictionary.")
        base_movies = current_movies
        new_movies = set()
        base_m2i = {f: i for i, f in enumerate(sorted(list(base_movies)))}

    elif base_movies and base_m2i:  # fine tune - ma wypluc zbior userów i u2i
        print("Operating in fine tuning mode.\nCreated new_movies set and updated base dictionary.")
        new_movies = current_movies - base_movies
        base_m2i.update({f: i for i, f in enumerate(sorted(list(new_movies)), start=len(base_m2i))})
    else:
        raise RuntimeError("Something failed while creating movies dictionaries and sets.")
    current_users.update(base_users)
    current_movies.update(base_movies)

    u2i: dict[str | int, int] = deepcopy(base_u2i)
    m2i: dict[str | int, int] = deepcopy(base_m2i)

    i2u: dict[int, str | int] = {i: u for u, i in u2i.items()}
    i2m: dict[int, str | int] = {i: m for m, i in m2i.items()}

    tokenized_data: list[tuple[int, int, float]] = []

    for u, f, r in data:
        tokenized_data.append((u2i[u], m2i[f], r))
    if len(new_movies) == 0 and len(new_users) == 0:
        print(
            f"\nTokenized {len(tokenized_data):7} records\nUsers count: {len(current_users):4}\nFilms count: {len(current_movies):4}")
    else:
        print(
            f"\nTokenized {len(tokenized_data):7} new records\nUsers count: {len(current_users):4} (+{len(new_users)})\nFilms count: {len(current_movies):4} (+{len(new_movies)})")
    return tokenized_data, u2i, i2u, m2i, i2m, current_users, current_movies

def train_val_test_split(data: list[tuple[int, int, float]],
                         train_share: float = 0.8,
                         val_share: float = 0.1,
                         test_share: float = 0.1,
                         shuffle: bool = True) -> tuple[
    list[tuple[int, int, float]], list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Splits the data set into train set, validation set and test set with given proportions."""
    if train_share + val_share + test_share != 1:
        raise ValueError("All the shares must sum up to 1.")

    if shuffle:
        random.shuffle(data)

    train_split = int(len(data) * train_share)
    val_split = int(len(data) * (train_share + val_share))

    return (data[:train_split],
            data[train_split:val_split],
            data[val_split:])