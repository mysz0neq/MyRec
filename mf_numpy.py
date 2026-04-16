"""(uid,fid,rtg)->wagi"""

import numpy as np
import tokenizer


class MF:
    def __init__(self,
                 data: list[tuple[int, int, float]],
                 dim: int):
        self.ld_ef = None
        self.ld_eu = None
        self.lr_b = None
        self.lr_e = None
        self.data = data
        self.users,self.movies=tokenizer.users_movies_sets(self.data)
        self.dim = dim
        self.user_matrix=np.random.normal(size=(len(self.users),self.dim),
                                          loc=0,
                                          scale=0.001)
        self.movie_matrix=np.random.normal(size=(len(self.movies),self.dim),
                                           loc=0,
                                           scale=0.001)
        self.movie_biases=np.zeros(shape=(len(self.movies),))
        self.user_biases=np.zeros(shape=(len(self.users),))
        self.mean=np.mean([r for _,_,r in self.data])

    def optimizer(self,
                  lr_e: float,
                  lr_b: float,
                  ld_eu: float,
                  ld_ef: float):
        self.lr_e=lr_e
        self.lr_b=lr_b
        self.ld_eu=ld_eu
        self.ld_ef=ld_ef

    def train(self,
              epochs:int):
        for epoch in range(epochs):
            loss=0
            for u,f,o in self.data:
                user_emb=self.user_matrix[u]
                movie_emb=self.movie_matrix[f]
                pred=np.dot(user_emb,movie_emb)+self.mean+self.user_biases[u]+self.movie_biases[f]
                error=pred-o
                loss+=error**2
                grad_u=2*user_emb*error+self.ld_eu*user_emb
                grad_f=2*movie_emb*error+self.ld_ef*movie_emb

                grad_bu=2*error
                grad_bf=2*error

                self.user_matrix[u]-=grad_u*self.lr_e
                self.movie_matrix[f]-=grad_f*self.lr_e
                self.user_biases[u]-=grad_bu*self.lr_b
                self.movie_biases[f]-=grad_bf*self.lr_b
            print(f"Epoch: {epoch} | Loss: {loss/len(self.data)}")

        return self.user_matrix,self.movie_matrix



