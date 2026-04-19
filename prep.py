"""Prepares data for training and testing"""
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