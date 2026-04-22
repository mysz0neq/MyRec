
class Film:
    def __init__(self,name:str,fid:int,number_of_ratings:int,average_rating:float):
        self.name=name
        self.id=fid
        self.num_ratings=number_of_ratings
        self.avg=average_rating
        self.runtime=None
        self.description=None
        self.genres=[]
    def __repr__(self):
        return f"Title: {self.name}\nDescription:{self.description}\n\nGenres: {self.genres}\nRuntime: {self.runtime}\n"
    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if not isinstance(other,Film):
            return False
        else:
            return self.id==other.id