from typing import Literal

from numpy.f2py.auxfuncs import isinteger

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
        means_u,means_f=filtr.means(self.data)
        for username,uid in self.u2i.items():
            self.i2u_obj[uid]=User(uid=uid,name=username,average_rating=means_u[uid])

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

    def recommendation(self,users_params:dict,how_much:int=20): #not deterministic??? for some reason?? ok figured it out: goto main.py:37
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
            print(u.id)
            print(u.name)
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
                else:
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
        df.sort_values(by=['Minimum rating'],inplace=True,ascending=False)
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

class UI:
    """Frontend class for handling API by user calls."""
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

        if choice==1: #Recommendation
            stats.print_df(self.recommendation(user=user))
            return self.logged_in(user=user)

        elif choice==2: #Group recommendation
            stats.print_df(self.group_recommendation(user=user))
            return self.logged_in(user=user)

        elif choice==3: #Changing lists
            raise NotImplementedError

        else: #Info about movie
            raise NotImplementedError
    def recommendation(self,user:User):
        print()
        print("What do you want to watch today?")
        print("1. Something good I haven't seen before...")
        print("2. Something from my watchlist...")
        print("3. I'll take anything, including rewatches...")
        print()
        choice = int(input("Pick a number from above: "))
        while choice not in range(1, 4):
            choice = int(input("Incorrect number. Pick one from above: "))
        if choice==1:
            settings={user:{
                'rewatches': 0,
                'pick_only_from': None,
                'exclude': user.not_interested,
                'min_predict': user.avg,
                'max_predict': None,
                'min_ratings': None,
                'max_ratings': None,
            }}
            return self.api.recommendation(settings,how_much=20)
        elif choice==2:
            settings={user:{
                'rewatches': 1,
                'pick_only_from': user.watchlist,
                'exclude': None,
                'min_predict': None,
                'max_predict': None,
                'min_ratings': None,
                'max_ratings': None,
            }}
            return self.api.recommendation(settings,how_much=20)
        else:
            settings={user:{
                'rewatches': 1,
                'pick_only_from': None,
                'exclude': None,
                'min_predict': None,
                'max_predict': None,
                'min_ratings': None,
                'max_ratings': None,
            }}
            return self.api.recommendation(settings,how_much=20)
    def group_recommendation(self,user:User):
        other_name=str(input("\nProvide other user username: "))
        while other_name not in self.api.u2i.keys():
            other_name = str(input("\nThere's no such user in the database.\nProvide username of someone that is in the database\n"))
        other_user=self.api.i2u_obj[self.api.u2i[other_name]]
        print(f"\nWhat do you and {other_name} want to watch today?")
        print("1. Something we both haven't seen before")
        print(f"2. Something I've seen before but {other_name} didn't")
        print(f"3. Something {other_name}'s seen before but I didn't")
        print("4. Something we both have seen before")
        print(f"5. Something that is both on mine and {other_name}'s watchlists")
        choice=int(input("Pick a number: "))
        while choice not in range(1,6):
            choice=int(input("Incorrect number. Pick one from above: "))
        if choice==1:
            settings = {
                user: {
                    'rewatches': 0,
                    'pick_only_from': None,
                    'exclude': user.not_interested,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
            },
                other_user: {
                    'rewatches': 0,
                    'pick_only_from': None,
                    'exclude': user.not_interested,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                }
            }
            return self.api.recommendation(settings,how_much=20)
        if choice==2:
            settings = {
                user: {
                    'rewatches': 2,
                    'pick_only_from': None,
                    'exclude': None,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                },
                other_user: {
                    'rewatches': 0,
                    'pick_only_from': None,
                    'exclude': user.not_interested,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                }
            }
            return self.api.recommendation(settings, how_much=20)
        if choice==3:
            settings = {
                user: {
                    'rewatches': 0,
                    'pick_only_from': None,
                    'exclude': user.not_interested,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                },
                other_user: {
                    'rewatches': 2,
                    'pick_only_from': None,
                    'exclude': None,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                }
            }
            return self.api.recommendation(settings, how_much=20)
        if choice==4:
            settings = {
                user: {
                    'rewatches': 2,
                    'pick_only_from': None,
                    'exclude': user.not_interested,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                },
                other_user: {
                    'rewatches': 2,
                    'pick_only_from': None,
                    'exclude': None,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                }
            }
            return self.api.recommendation(settings, how_much=20)
        else:
            settings = {
                user: {
                    'rewatches': 0,
                    'pick_only_from': user.watchlist,
                    'exclude': user.not_interested,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                },
                other_user: {
                    'rewatches': 0,
                    'pick_only_from': other_user.watchlist,
                    'exclude': user.not_interested,
                    'min_predict': None,
                    'max_predict': None,
                    'min_ratings': None,
                    'max_ratings': None,
                }
            }
            return self.api.recommendation(settings, how_much=20)
    def view_or_change_lists(self,user:User):
        users_lists_dict = dict(enumerate(user.lists.items()))

        print("Here are your lists:")
        print()
        for i,(list_name,content) in users_lists_dict.items():
            print(f"{i+1}. {list_name} ({len(content)} movies)")
        print()

        choice=int(input("Pick a number next to the list that you want to view/edit.\nPick 0 if you want to create a new list\n"))
        while choice not in range(len(users_lists_dict)+1):
            choice=int(input("Incorrect number. Pick one from above: "))
        print()
        if choice!=0:
            picked_list=users_lists_dict[choice-1]
            picked_list_name=picked_list[0]
            picked_list_content=list(picked_list[1])

            print("Here are the titles of movies on your list:")
            for i,movie in enumerate(picked_list_content):
                print(f"{i+1}. {movie.name}")
            print()
            print(f"What do you want to do with list '{picked_list_name}'?")
            print("1. View detailed content")
            print("2. Edit the list")
            print()

            action_choice=int(input("Pick a number from above: "))
            while action_choice not in [1,2]:
                action_choice=int(input("Incorrect number. Pick one from above: "))
            if action_choice==1:
                print(f"{picked_list_name}:")
                print()
                for i,movie in enumerate(picked_list_content):
                    print(f"{i+1}. {movie}\n")
            elif action_choice==2:
                print("What do you want to edit?")
                print("1. I want to remove some movies")
                print("2. I want to add some movies")
                print("3. I want to remove the list")
                print()
                edit_choice = int(input("Pick a number from above: "))
                while edit_choice not in [1, 2]:
                    edit_choice = int(input("Incorrect number. Pick a number from above: "))
                if edit_choice==1:
                    user.lists[picked_list_name]=self.remove_from_list(picked_list_content)
                elif edit_choice==2:
                    user.lists[picked_list_name]=self.add_to_list(picked_list_content)
                else:
                    del user.lists[picked_list_name]
        else:
            name = input("Create a name for the list: ")
            user.lists[name]=self.add_to_list([])
    def remove_from_list(self,users_list:list):
        to_remove=input("Enter numbers of movies that you want to delete from the list, separated by commas:\n").split(",")
        while not all([a.isdigit() for a in to_remove]):
            to_remove=input("Incorrect input.\nEnter numbers of movies that you want to delete from the list, separated by commas:\n").split(",")
        movies_to_remove=[]
        for number in to_remove:
            movie=users_list[int(number)-1]
            movies_to_remove.append(movie)
        for movie in movies_to_remove:
            users_list.remove(movie)
        print(f"Successfully deleted {len(to_remove)} movies.")
        return set(users_list)
    def add_to_list(self,users_list:list):
        users_set=set(users_list)
        to_add=input("Enter titles of movies that you want to add to the list, separated by commas:\n").split(",")
        while not all([(a in users_list) for a in users_set]):
            to_add=input("Incorrect input.\nEnter titles of movies that you want to add to the list, separated by commas:\n").split(",")
        for title in to_add:
            users_set.add(self.api.i2m_obj[self.api.m2i[title]])
        print(f"Successfully added {len(to_add)} movies.")
        return users_set
