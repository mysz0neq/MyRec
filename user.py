
class User:
    def __init__(self,name:str,uid:int,average_rating:float):
        self.name=name
        self.id=uid
        self.watched=set()
        self.ratings=dict()
        self.watchlist=set()
        self.not_interested=set()
        self.lists={'watchlist':self.watchlist}
        self.avg=average_rating
    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if not isinstance(other,User):
            return False
        else:
            return self.id==other.id