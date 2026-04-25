from os import PathLike
from typing import Literal
import pickle
import stats
import data_filter
import pandas as pd
from entity import User, Film
from model_training import MF


class API:
    """Backend class.

    The class is capable of:

    * giving recommendation to unlimited amount of users at once (group recommendations with the least misery sort)
    * creating movie rankings based on database stats: average ratings and number of ratings
    * saving and loading itself from or to a file
    """

    def __init__(self,
                 model: MF,
                 u2i: dict,
                 m2i: dict,
                 data: list[tuple[int, int, float]]):
        self.model = model
        self.u2i = u2i
        self.m2i = m2i
        self.data = data
        self.i2u_obj = dict()
        self.i2m_obj = dict()

        _, counter_f = data_filter.counters(self.data)
        means_u, means_f = data_filter.means(self.data)
        for username, uid in self.u2i.items():
            self.i2u_obj[uid] = User(uid=uid, name=username, average_rating=means_u[uid])

        for title, fid in self.m2i.items():
            self.i2m_obj[fid] = Film(fid=fid, name=title, number_of_ratings=counter_f[fid], average_rating=means_f[fid])

        for uid, fid, rtg in self.data:
            self.i2u_obj[uid].watched.add(self.i2m_obj[fid])
            self.i2u_obj[uid].ratings[fid] = rtg

    def recommendation(self,
                       users_params: dict[User, dict[str, int | set[Film] | float | None]],
                       how_much: int = 20) -> pd.DataFrame:
        """users_params format:\n
        {User: {rewatches: int[0 - not allowed, 1 - whatever, 2 - only rewatches]\n
                pick_only_from: set[Film] | None,\n
                exclude: set[Film] | None,\n
                min_predict: float | None,\n
                max_predict: float | None,\n
                min_ratings: int | None,\n
                max_ratings: int | None,\n
                }}\n
        There is no limit of users in users_params however the bigger the group the least
        probable it is that the method is going to return an acceptable amount of recommendations.\n
        \n
        It is assumed that format is correct in every way beacuse it is supposed to be created by the UI and not by the user.\n
        \n
        Notice that 'pick_only_from' and 'exclude' accept only one set
        which means that if user specifies couple of lists, then they are going to have to be merged into one set."""
        recs = dict()

        for u, p in users_params.items():
            rewatches_picks = set()
            only_from_picks = set()
            after_exclude_picks = set()
            min_predict_picks = set()
            max_predict_picks = set()
            min_ratings_picks = set()
            max_ratings_picks = set()
            for f in self.i2m_obj.values():
                predict = self.model.predict(u.id, f.id)
                if p['rewatches'] == 0:
                    if f not in u.watched:
                        rewatches_picks.add(f)
                elif p['rewatches'] == 1:
                    rewatches_picks.add(f)
                elif p['rewatches'] == 2:
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
                    if predict >= p['min_predict']:
                        min_predict_picks.add(f)

                if p['max_predict'] is None:
                    max_predict_picks.add(f)
                else:
                    if predict <= p['max_predict']:
                        max_predict_picks.add(f)

                if p['min_ratings'] is None:
                    min_ratings_picks.add(f)
                else:
                    if f.num_ratings >= p['min_ratings']:
                        min_ratings_picks.add(f)

                if p['max_ratings'] is None:
                    max_ratings_picks.add(f)
                else:
                    if f.num_ratings <= p['max_ratings']:
                        max_ratings_picks.add(f)

            picks = set.intersection(*[rewatches_picks,
                                       only_from_picks,
                                       after_exclude_picks,
                                       min_predict_picks,
                                       max_predict_picks,
                                       min_ratings_picks,
                                       max_ratings_picks])

            recs[u] = picks
        group_picks = [f for f in set.intersection(*recs.values())]

        list_for_df = []
        for f in group_picks:
            dic = dict()
            dic['Title'] = f.name
            dic['Average rating'] = round(f.avg, 2)
            dic['Number of ratings'] = f.num_ratings
            if f.runtime is not None:
                dic['Runtime'] = f.runtime
            predicted_ratings = []
            for u in users_params.keys():
                predict = self.model.predict(u.id, f.id)
                dic[f"{u.name}'s predicted rating"] = round(predict, 2)
                if f in u.watched:
                    dic[f"{u.name}'s real rating"] = u.ratings[f.id]
                predicted_ratings.append(predict)
            dic['Minimum rating'] = round(min(predicted_ratings), 2)
            list_for_df.append(dic)
        try:
            df = pd.DataFrame(list_for_df)
            df.sort_values(by=['Minimum rating'], inplace=True, ascending=False)
            df.reset_index(drop=True, inplace=True)
            df.index += 1
            df = df.fillna("-")
        except KeyError:
            return pd.DataFrame()
        return df.head(how_much).drop(['Minimum rating'], axis="columns")

    def ranking(self,
                by: Literal['avg', 'num'] = 'avg',
                ascending: bool = False,
                how_much: int = 100) -> pd.DataFrame:
        """Create movie database ranking sorted by average rating or number of ratings, ascending or descending.
        Returns a DataFrame"""
        df = pd.DataFrame([{'Title': f.name,
                            'Average rating': round(f.avg, 2),
                            'Number of ratings': f.num_ratings} for f in self.i2m_obj.values()])
        if by == 'avg' and ascending:
            df.sort_values(by=['Average rating'], ascending=True, inplace=True)
        elif by == 'avg' and not ascending:
            df.sort_values(by=['Average rating'], ascending=False, inplace=True)
        elif by == 'Number of ratings' and ascending:
            df.sort_values(by=['Number of ratings'], ascending=True, inplace=True)
        else:
            df.sort_values(by=["Number of ratings"], ascending=False, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.index += 1
        return df.head(how_much)

    def save(self, path):
        """Save API object, including model and vocabulary dicts to the file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> 'API':
        """Load API object from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class UI:
    """Frontend class for handling API by user.

    It's just for demo purposes since I'm not a frontend dev
    and, let's be honest, nobody is going to use console-based UI.


    The class allows user to:

    * manage their lists
    * get recommendations including group recommendations (for demo purposes only for pairs of users, sort by the least misery)
    * view info about movie database

    The class doesn't allow user to make any changes to the database like rating new movies
    or changing their ratings, because I won't do anything like that with sqlite, especially in the demo UI."""

    def __init__(self,
                 api_path: int | str | bytes | PathLike[str] | PathLike[bytes]):

        self.api_path = api_path
        self.api = API.load(api_path)
        self.main()

    def main(self):
        """To not get into account creation and DB schemas,
        let's assume that everyone is nice enough
        to not get into someone else's account :)"""

        print("Welcome to something something\n")
        username = str(input("Enter username: "))
        while username not in self.api.u2i.keys():
            username = str(input("Incorrect username. Provide correct one: "))
        user = self.api.i2u_obj[self.api.u2i[username]]
        self._logged_in(user)

    def _logged_in(self, user: User):
        print(f"\nHello {user.name}!\n\nWhat do you want to do?")
        print()
        print("1. Recommendation")
        print("2. Group recommendation")
        print("3. View and/or change my lists")
        print("4. Get info about something")
        print()
        choice = int(input("Pick a number from above: "))

        while choice not in range(1, 5):
            choice = int(input("Incorrect number. Pick one from above: "))

        if choice == 1:  # Recommendation
            stats.print_df(self._recommendation(user=user))
            self.api.save(self.api_path)
            return self._logged_in(user=user)

        elif choice == 2:  # Group recommendation
            stats.print_df(self._group_recommendation(user=user))
            self.api.save(self.api_path)
            return self._logged_in(user=user)

        elif choice == 3:  # Changing lists
            self._view_or_change_lists(user=user)
            self.api.save(self.api_path)
            return self._logged_in(user=user)
        else:  # Info about movie
            stats.print_df(self._info_about_movie(user=user))
            self.api.save(self.api_path)
            return self._logged_in(user=user)

    def _recommendation(self, user: User):
        print()
        print("What do you want to watch today?")
        print("1. Something good I haven't seen before...")
        print(f"2. Something from my watchlist ({len(user.watchlist)} movies)...")
        print("3. I'll take anything, including rewatches...")
        print("Enter without any input - return to main menu")
        print()
        choice = input("Pick a number from above: ")
        if not choice or not choice.isdigit():
            self._logged_in(user=user)
        else:
            choice = int(choice)
        while choice not in range(1, 4):
            choice = int(input("Incorrect number. Pick one from above: "))
        if choice == 1:
            settings = {user: {
                'rewatches': 0,
                'pick_only_from': None,
                'exclude': user.not_interested,
                'min_predict': user.avg,
                'max_predict': None,
                'min_ratings': None,
                'max_ratings': None,
            }}
            return self.api.recommendation(settings, how_much=20)
        elif choice == 2:
            settings = {user: {
                'rewatches': 1,
                'pick_only_from': user.watchlist,
                'exclude': None,
                'min_predict': None,
                'max_predict': None,
                'min_ratings': None,
                'max_ratings': None,
            }}
            return self.api.recommendation(settings, how_much=20)
        else:
            settings = {user: {
                'rewatches': 1,
                'pick_only_from': None,
                'exclude': None,
                'min_predict': None,
                'max_predict': None,
                'min_ratings': None,
                'max_ratings': None,
            }}
            return self.api.recommendation(settings, how_much=20)

    def _group_recommendation(self, user: User):
        other_name = str(input("\nProvide other user username: "))
        while other_name not in self.api.u2i.keys():
            other_name = str(
                input("\nThere's no such user in the database.\nProvide username of someone that is in the database\n"))
        other_user = self.api.i2u_obj[self.api.u2i[other_name]]
        print(other_user)
        print(f"\nWhat do you and {other_name} want to watch today?")
        print("1. Something we both haven't seen before")
        print(f"2. Something I've seen before but {other_name} didn't")
        print(f"3. Something {other_name}'s seen before but I didn't")
        print("4. Something we both have seen before")
        print(f"5. Something that is both on mine and {other_name}'s watchlists")
        print("Enter without any input - return to main menu")
        choice = input("Pick a number: ")
        if not choice or not choice.isdigit():
            self._logged_in(user=user)
        else:
            choice = int(choice)
        while choice not in range(1, 6):
            choice = int(input("Incorrect number. Pick one from above: "))
        if choice == 1:
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
            return self.api.recommendation(settings, how_much=20)
        if choice == 2:
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
        if choice == 3:
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
        if choice == 4:
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

    def _view_or_change_lists(self, user: User):
        users_lists_dict = dict(enumerate(user.lists.items()))

        print("Here are your lists:")
        print()
        for i, (list_name, content) in users_lists_dict.items():
            print(f"{i + 1}. {list_name} ({len(content)} movies)")
        print()

        choice = input(
            f"Pick a number next to the list that you want to view/edit.\nPress enter without any input if you want to create a new list or {len(users_lists_dict) + 2} if you want to return to main menu\n")
        if not choice or not choice.isdigit():
            self._logged_in(user=user)
        else:
            choice = int(choice)
        while choice not in range(len(users_lists_dict) + 1):
            choice = int(input("Incorrect number. Pick one from above: "))
        print()
        if choice != 0:
            picked_list = users_lists_dict[choice - 1]
            picked_list_name = picked_list[0]
            picked_list_content = picked_list[1]
            temp_picked_list_content = list(picked_list_content)

            print("Here are the titles of movies on your list:")
            for i, movie in enumerate(temp_picked_list_content):
                print(f"{i + 1}. {movie.name}")
            print()
            print(f"What do you want to do with list '{picked_list_name}'?")
            print("1. View detailed content")
            print("2. Edit the list")
            print("Enter without any input - return to main menu")
            print()

            action_choice = input("Pick a number from above: ")
            if not action_choice or not action_choice.isdigit():
                self._logged_in(user=user)
            else:
                action_choice = int(action_choice)
            while action_choice not in [1, 2]:
                action_choice = int(input("Incorrect number. Pick one from above: "))
            if action_choice == 1:
                print(f"{picked_list_name}:")
                print()
                for i, movie in enumerate(temp_picked_list_content):
                    print(f"{i + 1}. {movie}\n")
            elif action_choice == 2:
                print("What do you want to edit?")
                print("1. I want to remove some movies")
                print("2. I want to add some movies")
                print("3. I want to remove the list")
                print("Enter without any input - return to main menu")
                print()
                edit_choice = input("Pick a number from above: ")
                if not edit_choice or not edit_choice.isdigit():
                    self._logged_in(user=user)
                else:
                    edit_choice = int(edit_choice)
                while edit_choice not in [1, 2, 3]:
                    edit_choice = int(input("Incorrect number. Pick a number from above: "))
                if edit_choice == 1:
                    self._remove_from_list(users_set=picked_list_content, user=user)
                elif edit_choice == 2:
                    self._add_to_list(users_set=picked_list_content, user=user)
                else:
                    del user.lists[picked_list_name]
        else:
            name = input("Create a name for the list: ")
            user.lists[name] = set()
            self._add_to_list(user=user, users_set=user.lists[name])

    def _remove_from_list(self,
                          user: User,
                          users_set: set):

        to_remove = (input("Enter numbers of movies that you want to delete from the list, separated by commas\n"
                           "Press enter without any input to return to main menu:\n")
                     .split(","))
        if not to_remove:
            self._logged_in(user=user)
        while not all([a.isdigit() for a in to_remove]):
            to_remove = (input("Incorrect input.\n"
                               "Enter numbers of movies that you want to delete from the list, separated by commas:\n")
                         .split(","))
        users_list = list(users_set)
        for number in to_remove:
            movie = users_list[int(number) - 1]
            users_set.remove(movie)
        print(f"Successfully deleted {len(to_remove)} movies.")
        return None

    def _add_to_list(self, user: User, users_set: set):
        to_add = (input("Enter titles of movies that you want to add to the list, separated by commas\n"
                        "Press enter without any input to return to main menu:\n")
                  .split(","))
        if not to_add:
            self._logged_in(user=user)
        users_list = list(users_set)
        while not all([(a in users_list) for a in users_set]):
            to_add = input(
                "Incorrect input.\nEnter titles of movies that you want to add to the list, separated by commas:\n").split(
                ",")
        for title in to_add:
            users_set.add(self.api.i2m_obj[self.api.m2i[title.strip()]])
        print(f"Successfully added {len(to_add)} movies.")
        return None

    def _info_about_movie(self, user: User):
        print("\nWhat do you want to view?")
        print("1. Top 100 highest rated movies.")
        print("2. Top 100 movies with the most ratings.")
        print("3. Info about specific title")
        print("Enter without any input - return to main menu")
        choice = input("\nPick a number from above: ")
        if not choice or not choice.isdigit():
            self._logged_in(user=user)
        else:
            choice = int(choice)
        while choice not in range(1, 4):
            choice = int(input("\nIncorrect number. Pick one from above: "))

        if choice == 1:
            return self.api.ranking(by='avg', ascending=False)
        elif choice == 2:
            return self.api.ranking(by='num', ascending=False)
        else:
            title = input("\nWhat movie do you want to get info about?\n")
            while title not in self.api.m2i.keys():
                title = input("\nThere's no such movie in the database.\nPick a movie from the database: ")
            movie = self.api.i2m_obj[self.api.m2i[title]]
            return movie
