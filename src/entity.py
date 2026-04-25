class User:
    """Entity class for user. Holds info about:

    * their ratings
    * watched movies
    * watchlist
    * not interested list
    * custom-made lists
    * average rating
    * ID given by the tokenizer
    * username"""
    def __init__(self,
                 name: str,
                 uid: int,
                 average_rating: float):
        self.name = name
        self.id = uid
        self.watched = set()
        self.ratings = dict()
        self.watchlist = set()
        self.not_interested = set()
        self.lists = {'watchlist': self.watchlist, 'not interested':self.not_interested}
        self.avg = average_rating

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self,
               other) -> bool:
        if not isinstance(other, User):
            return False
        else:
            return self.id == other.id

class Film:
    """Entity class for a film. Holds info about:

    * number of ratings of the movie
    * average rating
    * title
    * ID given by the tokenizer
    * runtime, description and genres TBA from TMDb

    __repr__ method gives all of this information in the formatted string."""
    def __init__(self,
                 name: str,
                 fid: int,
                 number_of_ratings: int,
                 average_rating: float):
        self.name = name
        self.id = fid
        self.num_ratings = number_of_ratings
        self.avg = average_rating
        self.runtime = None
        self.description = None
        self.genres = []

    def __repr__(self) -> str:
        return f"Title: {self.name}\nDescription: {self.description}\n\nGenres: {self.genres}\nRuntime: {self.runtime}\n"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self,
               other) -> bool:
        if not isinstance(other, Film):
            return False
        else:
            return self.id == other.id
