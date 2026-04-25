"""Data analysis tools. Kind of a mess. FIXME"""
from typing import Any, Literal
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import warnings
import matplotlib.pyplot as plt
import data_filter


class Analyzable:
    """Object that has ratings which can be analyzed."""
    def __init__(self,
                 ratings: defaultdict[int | str, float],
                 name: int | str):
        self.name = name
        self.ratings = [i for i in ratings.values()]
        self.num_ratings = len(self.ratings)

    def mean(self) -> float:
        """Returns mean average value of ratings."""
        return float(np.mean(self.ratings))

    def median(self) -> float:
        """Returns median value of ratings."""
        return float(np.median(self.ratings))

    def distribution(self) -> list[tuple[float, int]]:
        """Returns list of tuples where first element is a rating equivalence class (e.g. 7/10)
        and second element is number of these ratings in the data."""
        return [(r, c) for r, c in Counter(self.ratings).items()]

    def variance(self) -> float:
        """Returns variance of ratings."""
        return float(np.var(self.ratings))

    def centile_distribution(self,
                             centile: int | tuple[int, ...] | list[int]) -> list[tuple[float, ...]]:
        """For each centile specified (scale 0-100), a distribution of ratings is returned.

        Example:
        40% of all ratings (sorted by the most common rating class to the least common) are 10/10, 70% of all ratings are 10/10 and 1/10, and 100% of all ratings is a full 1-10 scale. Then:

        * for centiles 1-39 returned value would be ().
        * for centiles 40-69 returned value would be (10).
        * for centiles 70-99 returned value would be (1,10).
        * for centiles 100 returned value would be (1,2,3,4,5,6,7,8,9,10).

        You can specify multiple centile values and in return get distribution for each one.

        Note: no matter the number of centile values specified, the returned type is always a list."""
        distribution = sorted(self.distribution(), key=lambda x: x[1], reverse=True)
        centiles_ratings = []
        if isinstance(centile, int):
            centile_list = [centile]
        else:
            centile_list = centile
        for cent in centile_list:
            ratings = []
            control_sum = 0
            it = iter(distribution)
            while control_sum < cent * self.num_ratings / 100:
                r, count = next(it)
                ratings.append(r)
                control_sum += count
            centiles_ratings.append(tuple(sorted(ratings)))
        return centiles_ratings


class Stats:
    """Data science tool for the dataset."""
    def __init__(self,
                 data: list[tuple[int|str, int|str, float]]) -> None:
        self.df = None
        self.data = data
        self.is_single_centile = True

    def get_info(self,
                 mode: Literal['u','m'],
                 centile: int | tuple[int, ...] = 80,
                 i2u: dict = None,
                 sort_by: Literal['num_ratings','avg','median','var','spread','name'] = 'num_ratings',
                 ascending: bool = False) -> pd.DataFrame:
        """Creates a DataFrame of users or movies (hence the mode). Every row holds info about:

        * username/movie title
        * number of ratings given by a user/to a movie
        * average rating
        * median of the ratings
        * variance of the ratings
        * centile-wise ratings distribution
        * spread of the centile-wise ratings distribution

        The DataFrame is sorted as specified in the method arguments.

        i2u dictionary is necessary if data given in the constructor is already tokenized.

        Note: the last two columns (i.e. centile-wise ratings distribution and spread of this distribution)
        are either lists or single-values/tuples depending on the number of centile values specified in the parameters of this method.
        If more than one centile value is given, the spread column is absent in the correlation matrix!"""
        self.is_single_centile = False
        if isinstance(centile, int):
            self.is_single_centile = True
        elif isinstance(centile, (tuple, list)) and len(centile) == 1:
            self.is_single_centile = True

        if not self.is_single_centile:
            warnings.warn("With multiple centile values some functions may be unavailable.")
        list_of_analyzables = []
        if not mode:
            raise ValueError("You have to specify a mode")

        ratings_dict: defaultdict[int | str, defaultdict[int | str, float]] = defaultdict(lambda: defaultdict(float))
        if mode == "u":
            users_set = set()
            for u, m, r in self.data:
                ratings_dict[u][m] = r
                users_set.add(u)
            for u in users_set:
                list_of_analyzables.append(Analyzable(ratings_dict[u], u))
        elif mode == "m":
            movies_set = set()
            for u, m, r in self.data:
                ratings_dict[m][u] = r
                movies_set.add(m)
            for m in movies_set:
                list_of_analyzables.append(Analyzable(ratings_dict[m], m))
        else:
            raise Exception("Mode must be 'u' or 'm'")
        data_rows = []

        for o in list_of_analyzables:
            cent_vals = o.centile_distribution(centile)
            if self.is_single_centile:
                spread = len(cent_vals[0])
            else:
                spread = [len(r) for r in cent_vals]
            data_rows.append({
                "Name": o.name if not i2u else i2u[o.name],
                "Number of ratings": o.num_ratings,
                "Mean": round(o.mean(), 2),
                "Median": o.median(),
                "Variance": round(o.variance(), 2),
                "Centile-wise distribution": cent_vals if not self.is_single_centile else cent_vals[0],
                "Spread of ratings": spread,
            })

        sorting_map = {
            'num_ratings': 'Number of ratings',
            'avg': 'Mean',
            'median': 'Median',
            'var': 'Variance',
            'spread': 'Spread of ratings',
            'name': 'Name'
        }
        if sort_by not in sorting_map:
            warnings.warn(f"{sort_by} is not a valid sorting method. DataFrame sorted by default method.")
        self.df = (pd.DataFrame(data_rows)
                   .sort_values(by=sorting_map.get(sort_by, 'Number of ratings'),
                                ascending=ascending)
                   .reset_index(drop=True))

        return self.df

    def correlation_matrix(self,
                           mode: Literal['u','m']) -> pd.DataFrame:
        """Returns correlation matrix of the DataFrame created by get_info method.

        Note: If more than one centile value is given in get_info method,
        the spread column is absent in the correlation matrix!"""
        if self.df is None:
            self.df = self.get_info(mode=mode)
        if not self.is_single_centile:
            warnings.warn("Dropped 'Spread of ratings' column since user provided multiple centile values.")
        return (self.df
                .drop(columns=["Name", "Centile-wise distribution"])
                .corr(numeric_only=True)).round(4) #spread column is auto dropped if it's a tuple because of numeric_only flag


def print_df(df: pd.DataFrame) -> None:
    """Used for printing full sized DataFrame."""
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.precision', 10,
                           ):
        print(df)
    return None


def baseline_loss(train_set:list,
                  val_set:list) -> float:
    """Returns baseline loss value of the set without any machine learning involved.
    Computed by assuming that every user gives movies the same rating which is user's average rating.
    Basically this method returns weighted average user's ratings variation of the dataset, weighted by number of records of each user"""
    counter_users, _ = data_filter.counters(train_set)
    user_sums = defaultdict(int)
    if len(train_set) != 0:
        global_mean = sum(o for _, _, o in train_set) / len(train_set)
    else:
        global_mean = 0
    for u, _, o in train_set:
        user_sums[u] += o

    avg_user = {u: user_sums[u] / counter_users[u] for u in user_sums}

    loss = 0
    for u, _, o in val_set:
        loss += (o - avg_user.get(u, global_mean)) ** 2
    return loss / len(val_set)


def plots(history:dict[str, Any]) -> None:
    """Creates plots out of provided history dictionary alongside the differences plots of this data."""
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 28))

    ax[0, 0].set_title("Loss curves")
    ax[0, 0].set_ylabel("Loss value")
    ax[0, 0].plot(history['x_labels'], history['train_labels'], label="Train loss")
    ax[0, 0].plot(history['x_labels'], history['val_labels'], label="Validation loss")
    ax[0, 0].axhline(history['test_loss'], linestyle='--', label="Test loss")
    ax[0, 0].axhline(history['baseline_loss'], linestyle='--', label="Baseline loss")

    ax[1, 0].set_title("Biases curves")
    ax[1, 0].set_ylabel("Bias avg. value")
    ax[1, 0].plot(history['x_labels'], history['users_biases'], label="Users biases")
    ax[1, 0].plot(history['x_labels'], history['films_biases'], label="Films biaes")

    ax[2, 0].set_title("Embeddings norms curves")
    ax[2, 0].set_ylabel("Embedding norm avg. value")
    ax[2, 0].plot(history['x_labels'], history['users_vectors'], label="Users embeddings norms")
    ax[2, 0].plot(history['x_labels'], history['films_vectors'], label="Films embeddings norms")

    ax[3, 0].set_title("Absolute values of biases curves")
    ax[3, 0].set_ylabel("Bias abs. avg. value")
    ax[3, 0].plot(history['x_labels'], history['users_abs_biases'], label="Absolute values of users biases")
    ax[3, 0].plot(history['x_labels'], history['films_abs_biases'], label="Absolute values of films biases")

    ###DERIVATIVES
    ax[0, 1].set_title("Loss curves difference")
    ax[0, 1].set_ylabel("Loss difference value")
    ax[0, 1].plot(history['x_labels'], difference_fun(history['train_labels']), label="Train loss")
    ax[0, 1].plot(history['x_labels'], difference_fun(history['val_labels']), label="Validation loss")

    ax[1, 1].set_title("Biases curves difference")
    ax[1, 1].set_ylabel("Bias avg. difference value")
    ax[1, 1].plot(history['x_labels'], difference_fun(history['users_biases']), label="Users biases")
    ax[1, 1].plot(history['x_labels'], difference_fun(history['films_biases']), label="Films biaes")

    ax[2, 1].set_title("Embeddings norms curves difference")
    ax[2, 1].set_ylabel("Embedding norm avg. difference value")
    ax[2, 1].plot(history['x_labels'], difference_fun(history['users_vectors']), label="Users embeddings norms")
    ax[2, 1].plot(history['x_labels'], difference_fun(history['films_vectors']), label="Films embeddings norms")

    ax[3, 1].set_title("Absolute values of biases curves difference")
    ax[3, 1].set_ylabel("Bias abs. avg. difference value")
    ax[3, 1].plot(history['x_labels'], difference_fun(history['users_abs_biases']),
                  label="Absolute values of users biases")
    ax[3, 1].plot(history['x_labels'], difference_fun(history['films_abs_biases']),
                  label="Absolute values of films biases")

    for a in ax.flat:
        a.set_xlabel("Epoch")
        a.axvline(history['best_loss_epoch'], lw=1.5, linestyle=(0, (5, 10)), label="Best loss epoch",
                  color='red')
        for point in history['lr_lowering_points']:
            a.axvline(point, linestyle='solid', alpha=0.8, lw=0.6, color='green')
        a.legend()
    plt.show()
    return None

def difference_fun(y: list):
    """Basic difference function for a list. Starting value is 0."""
    return np.insert(np.diff(y), 0, 0)
