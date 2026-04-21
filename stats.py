"""Data analysis tools.
TODO: best loss point in plots, points when learning rate got lowered, derivatives"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import warnings
import matplotlib.pyplot as plt
import filtr

class Analyzable:
    def __init__(self,
                 ratings:defaultdict[int|str,float],
                 name:int|str):
        self.name=name
        self.ratings = [i for i in ratings.values()]
        self.num_ratings = len(self.ratings)

    def mean(self):
        return np.mean(self.ratings)
    def median(self) -> np.float64:
        return np.median(self.ratings)
    def distribution(self) -> list[tuple[float,int]]:
        return [(r,c) for r,c in Counter(self.ratings).items()]
    def variance(self):
        return np.var(self.ratings)

    def centile_ratings(self,
                        centile:int|tuple[int,...]|list[int]) -> list[tuple[float,...]]:

        distribution = sorted(self.distribution(),key=lambda x: x[1],reverse=True)
        centiles_ratings=[]
        if isinstance(centile,int):
            centile_list=[centile]
        else:
            centile_list=centile
        for cent in centile_list:
            ratings = []
            control_sum = 0
            it = iter(distribution)
            while control_sum<cent*self.num_ratings/100:
                r,count=next(it)
                ratings.append(r)
                control_sum+=count
            centiles_ratings.append(tuple(sorted(ratings)))
        return centiles_ratings

class Stats:
    def __init__(self,
                 data:list[tuple[int,int,float]]) -> None:
        self.df = None
        self.data=data
        self.is_single_centile=True

    def get_info(self,
                 mode: str,
                 centile:int|tuple[int,...]=80,
                 i2u:dict=None,
                 sort_by:str='num_ratings',
                 ascending:bool=False) -> pd.DataFrame:
        self.is_single_centile=False
        if isinstance(centile,int):
            self.is_single_centile=True
        elif isinstance(centile,(tuple,list)) and len(centile)==1:
            self.is_single_centile=True

        if not self.is_single_centile:
            warnings.warn("With multiple centile values some functions may be unavailable.")
        list_of_analyzables=[]
        if not mode:
            raise Exception("You have to specify a mode")

        ratings_dict:defaultdict[int|str,defaultdict[int|str,float]]=defaultdict(lambda: defaultdict(float))
        if mode=="u":
            users_set=set()
            for u,m,r in self.data:
                ratings_dict[u][m]=r
                users_set.add(u)
            for u in users_set:
                list_of_analyzables.append(Analyzable(ratings_dict[u],u))
        elif mode=="m":
            movies_set=set()
            for u,m,r in self.data:
                ratings_dict[m][u]=r
                movies_set.add(m)
            for m in movies_set:
                list_of_analyzables.append(Analyzable(ratings_dict[m],m))
        else:
            raise Exception("Mode must be 'u' or 'm'")
        data_rows=[]

        for o in list_of_analyzables:
            cent_vals=o.centile_ratings(centile)
            if self.is_single_centile:
                spread=len(cent_vals[0])
            else:
                spread=[len(r) for r in cent_vals]
            data_rows.append({
                "Name": o.name if not i2u else i2u[o.name],
                "Number of ratings": o.num_ratings,
                "Mean": round(o.mean(),2),
                "Median": o.median(),
                "Variance": round(o.variance(),2),
                "Centile": cent_vals if not self.is_single_centile else cent_vals[0],
                "Spread of ratings": spread,
            })

        sorting_map={
            'num_ratings':'Number of ratings',
            'avg':'Mean',
            'median':'Median',
            'var':'Variance',
            'spread':'Spread of ratings',
            'name':'Name'
        }
        if sort_by not in sorting_map:
            warnings.warn(f"{sort_by} is not a valid sorting method. DataFrame sorted by default method.")
        self.df=(pd.DataFrame(data_rows)
                 .sort_values(by=sorting_map.get(sort_by,'Number of ratings'),
                              ascending=ascending)
                 .reset_index(drop=True))

        return self.df

    def correlation_matrix(self,
                           mode:str):
        if self.df is None:
            self.get_info(mode=mode)
        if not self.is_single_centile:
            warnings.warn("Dropped 'Spread of ratings' column since user provided multiple centile values.")
        return (self.df
                .drop(columns=["Name","Centile"])
                .corr(numeric_only=True)).round(4)

def print_df(df:pd.DataFrame) -> None:
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.precision', 10,
                           ):
        print(df)
    return None

def baseline_loss(train_set,val_set):
    counter_users,_=filtr.counters(train_set)
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

def plots(history):
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

    ax[0, 1].set_title("Embeddings norms curves")
    ax[0, 1].set_ylabel("Embedding norm avg. value")
    ax[0, 1].plot(history['x_labels'], history['users_vectors'], label="Users embeddings norms")
    ax[0, 1].plot(history['x_labels'], history['films_vectors'], label="Films embeddings norms")

    ax[1, 1].set_title("Absolute values of biases curves")
    ax[1, 1].set_ylabel("Bias abs. avg. value")
    ax[1, 1].plot(history['x_labels'], history['users_abs_biases'], label="Absolute values of users biases")
    ax[1, 1].plot(history['x_labels'], history['films_abs_biases'], label="Absolute values of films biases")

    ###DERIVATIVES
    ax[2, 0].set_title("Loss curves difference")
    ax[2, 0].set_ylabel("Loss difference value")
    ax[2, 0].plot(history['x_labels'], difference_fun(history['train_labels']), label="Train loss")
    ax[2, 0].plot(history['x_labels'], difference_fun(history['val_labels']), label="Validation loss")

    ax[3, 0].set_title("Biases curves difference")
    ax[3, 0].set_ylabel("Bias avg. difference value")
    ax[3, 0].plot(history['x_labels'], difference_fun(history['users_biases']), label="Users biases")
    ax[3, 0].plot(history['x_labels'], difference_fun(history['films_biases']), label="Films biaes")

    ax[2, 1].set_title("Embeddings norms curves difference")
    ax[2, 1].set_ylabel("Embedding norm avg. difference value")
    ax[2, 1].plot(history['x_labels'], difference_fun(history['users_vectors']), label="Users embeddings norms")
    ax[2, 1].plot(history['x_labels'], difference_fun(history['films_vectors']), label="Films embeddings norms")

    ax[3, 1].set_title("Absolute values of biases curves difference")
    ax[3, 1].set_ylabel("Bias abs. avg. difference value")
    ax[3, 1].plot(history['x_labels'], difference_fun(history['users_abs_biases']), label="Absolute values of users biases")
    ax[3, 1].plot(history['x_labels'], difference_fun(history['films_abs_biases']), label="Absolute values of films biases")

    for a in ax.flat:
        a.set_xlabel("Epoch")
        a.axvline(history['best_loss_epoch'], lw=1.5, linestyle=(0, (5, 10)), label="Best loss epoch",
                         color='red')
        for point in history['lr_lowering_points']:
            a.axvline(point, linestyle='solid', alpha=0.8, lw=0.6, color='green')
        a.legend()
    plt.show()

def difference_fun(y:list):
    return np.insert(np.diff(y),0,0)