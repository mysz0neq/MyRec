from typing import Literal
import filtr
import pandas as pd

import stats
from user import User
from film import Film
from mf import MF
class API:
    def __init__(self,model:MF, u2i:dict,m2i:dict,data:list[tuple[int,int,float]]):
        self.model=model
        self.u2i=u2i
        self.m2i=m2i
        self.data=data
        self.i2u_obj=dict()
        self.i2m_obj=dict()
        _,counter_f=filtr.counters(self.data)
        _,means_f=filtr.means(self.data)
        for username,uid in self.u2i.items():
            self.i2u_obj[uid]=User(uid=uid,name=username)

        for title,fid in self.m2i.items():
            self.i2m_obj[fid]=Film(fid=fid,name=title,number_of_ratings=counter_f[fid],average_rating=means_f[fid])

        for uid,fid,rtg in self.data:
            self.i2u_obj[uid].watched.add(self.i2m_obj[fid])
            self.i2u_obj[uid].ratings[fid]=rtg

    def create_list(self,uid:int,list_name:str) -> None:
        user=self.i2u_obj[uid]
        user.lists[list_name]=set()
        print(f"Successfully created list {list_name} for user {user.name}")
        return None

    def alter_list(self,uid:int,list_name:str,movie:Film,add:bool):
        """add parameter is True for appending and False for removing"""
        user=self.i2u_obj[uid]
        users_list=user.lists[list_name]
        if add:
            users_list.add(movie)
        else:
            try:
                users_list.remove(movie)
            except KeyError:
                print(f"There's no {movie.name} movie in the list!")

    def recommendation(self,users_params:dict,how_much:int=20):
        """Params format:
        {User: {
                rewatches: int[0 - not allowed, 1 - whatever, 2 - only rewatches]
                pick_only_from: set[Film] | None,
                exclude: set[Film] | None,
                min_predict: float | None,
                max_predict: float | None,
                min_ratings: int | None,
                max_ratings: int | None,
                }}
        It is assumed that format is correct in every way"""
        recs=dict()
        for u,p in users_params.items():
            rewatches_picks=set()
            only_from_picks=set()
            after_exclude_picks=set()
            min_predict_picks=set()
            max_predict_picks=set()
            min_ratings_picks=set()
            max_ratings_picks=set()
            for f in self.i2m_obj.values():
                predict=self.model.predict(u.id,f.id)
                if p['rewatches']==0:
                    if f not in u.watched:
                        rewatches_picks.add(f)
                elif p['rewatches']==1:
                    rewatches_picks.add(f)
                elif p['rewatches']==2:
                    if f in u.watched:
                        rewatches_picks.add(f)

                if p['pick_only_from'] is None:
                    only_from_picks.add(f)
                else:
                    if f in p['pick_only_from']:
                        only_from_picks.add(f)

                if p['exclude'] is not None:
                    if f not in p['exclude']:
                        after_exclude_picks.add(f)

                if p['min_predict'] is None:
                    min_predict_picks.add(f)
                else:
                    if predict>=p['min_predict']:
                        min_predict_picks.add(f)

                if p['max_predict'] is None:
                    max_predict_picks.add(f)
                else:
                    if predict<=p['max_predict']:
                        max_predict_picks.add(f)

                if p['min_ratings'] is None:
                    min_ratings_picks.add(f)
                else:
                    if f.num_ratings>=p['min_ratings']:
                        min_ratings_picks.add(f)

                if p['max_ratings'] is None:
                    max_ratings_picks.add(f)
                else:
                    if f.num_ratings<=p['max_ratings']:
                        max_ratings_picks.add(f)

            picks=set.intersection(*[rewatches_picks,
                                     only_from_picks,
                                     after_exclude_picks,
                                     min_predict_picks,
                                     max_predict_picks,
                                     min_ratings_picks,
                                     max_ratings_picks])

            recs[u]=picks
        group_picks=[f for f in set.intersection(*recs.values())]

        list_for_df=[]
        for f in group_picks:
            dic=dict()
            dic['Title']=f.name
            dic['Average rating']=f.avg
            dic['Number of ratings']=f.num_ratings
            dic['Runtime']=f.runtime
            predicted_ratings=[]
            for u in users_params.keys():
                predict=self.model.predict(u.id,f.id)
                dic[f'{u.name} predicted rating']=predict
                if f in u.watched:
                    dic[f'{u.name} real rating']=u.ratings[f.id]
                predicted_ratings.append(predict)
            dic['Minimum rating']=min(predicted_ratings)
            list_for_df.append(dic)
        df=pd.DataFrame(list_for_df)
        df.sort_values(by=['Minimum rating'],inplace=True)
        df.reset_index(drop=True,inplace=True)
        df.index+=1
        return df.head(how_much)

    def ranking(self,by:Literal['avg','num']='avg',ascending:bool=False):
        df=pd.DataFrame([{'Title': f.name,
                          'Average rating': f.avg,
                          'Number of ratings':f.num_ratings} for f in self.i2m_obj.values()])
        if by=='avg' and ascending:
            df.sort_values(by=['Average rating'],ascending=True,inplace=True)
        elif by=='avg' and not ascending:
            df.sort_values(by=['Average rating'],ascending=False,inplace=True)
        elif by=='Number of ratings' and ascending:
            df.sort_values(by=['Number of ratings'],ascending=True,inplace=True)
        else:
            df.sort_values(by=["Number of ratings"],ascending=False,inplace=True)
        return df.head(100)

class UI: #TODO
    """Frontend class for handling API by user calls - it makes sure that every API call is correctly formatted etc."""
    def __init__(self,api:API):
        self.api=api
        self.main()
    def main(self):
        """To not get into account creation and DB schemas,
        let's assume that every username is unique,
        and everyone is nice enough to not get into someone else's account :)"""

        print("Welcome to something something\n")
        username=str(input("Enter username: "))
        while username not in self.api.u2i.keys():
            username = str(input("Incorrect username. Provide correct one: "))
        user=self.api.i2u_obj[self.api.u2i[username]]
        self.logged_in(user)
    def logged_in(self,user:User):
        print(f"\nHello {user.name}!\n\nWhat do you want to do?")
        print()
        print("1. Recommendation")
        print("2. Group recommendation")
        print("3. View and/or change my lists")
        print("4. Get info about a movie")
        print()
        choice=int(input("Pick a number from above: "))
        while choice not in range(1,5):
            choice=int(input("Incorrect number. Pick one from above: "))
        if choice==1:
            raise NotImplementedError
        elif choice==2:
            raise NotImplementedError
        elif choice==3:
            raise NotImplementedError
        elif choice==4:
            raise NotImplementedError
