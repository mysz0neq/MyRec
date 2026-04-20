"""Gets data ready for analysis and Machine Learning"""

def users_movies_sets(data:list[tuple[str|int,str|int,float]]) -> tuple[set,set]:
    """Returns sets of users and movies"""
    users=set()
    movies=set()
    for u,f,_ in data:
        users.add(u)
        movies.add(f)
    return users,movies

def tokenizer(data: list[tuple[str,str,float]],
              users:set=None,
              movies:set=None) -> tuple[list[tuple[int,int,float]],dict,dict,dict,dict]:
    """Changes users and movies names to IDs and returns the dictionaries."""
    if not users or not movies:
        users,movies=users_movies_sets(data)

    u2i={u:i for i,u in enumerate(users)}
    i2u={i:u for i,u in enumerate(users)}
    m2i={m:i for i,m in enumerate(movies)}
    i2m={i:m for i,m in enumerate(movies)}

    new_data=[]

    for u,f,r in data:
        new_data.append((u2i[u],m2i[f],r))

    return new_data,u2i,i2u,m2i,i2m

import random

def train_val_test_split(data:list[tuple[int,int,float]],
                         train_share:float=0.8,
                         val_share:float=0,
                         test_share:float=0.2,
                         shuffle:bool=True) -> tuple[list[tuple[int,int,float]],list[tuple[int,int,float]],list[tuple[int,int,float]]]:
    if train_share+val_share+test_share!=1:
        raise ValueError("All the shares must sum up to 1.")

    if shuffle:
        random.shuffle(data)

    train_split=int(len(data)*train_share)
    val_split=int(len(data)*(train_share+val_share))

    return (data[:train_split],
            data[train_split:val_split],
            data[val_split:])