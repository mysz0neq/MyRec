"""Cleans data"""

from collections import defaultdict

def iteration(data: list[tuple[str|int,str|int,float]],
              min_u:int,
              min_f:int) -> list[tuple[str|int,str|int,float]]:
    """Single iteration of K-core filtering algorithm"""
    counter_u, counter_f = counters(data)

    new_data=[]
    for u,f,r in data:
        if counter_u[u]>=min_u and counter_f[f]>=min_f:
            new_data.append((u,f,r))
    return new_data

def counters(data: list[tuple[str|int,str|int,float]]) -> tuple[defaultdict[str|int,float],defaultdict[str|int,float]]:
    """Returns counter dictionaries with number of records for each user and movie"""

    counter_u=defaultdict(int)
    counter_f=defaultdict(int)
    for u,f,_ in data:
        counter_u[u]+=1
        counter_f[f]+=1
    return counter_u,counter_f

def filtr(data: list[tuple[str|int,str|int,float]],
          min_u:int,
          min_f:int) -> list[tuple[str|int,str|int,float]]:
    """Implements K-core filtering algorithm to clean
    data records from users with less than min_u
    ratings and movies less than min_f ratings"""

    previous=0
    current=len(data)

    while previous!=current:
        data=iteration(data,min_u,min_f)
        previous=current
        current=len(data)
    return data