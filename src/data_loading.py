"""Introduces data to the pipeline."""

import sqlite3
from os import PathLike

def get_data(path: str | bytes | PathLike[str] | PathLike[bytes]) -> list[tuple[str, str, float]]:
    """This method is used for parsing data. Requirements:

    1. sqlite database
    2. Data is in table named 'oceny' (ratings in polish)
    3. Data is in the form of (username, movie title, rating)

    If your data is different, you have to edit this method or create your own."""
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''SELECT *
                      FROM oceny''')
    data = cursor.fetchall()
    cursor.close()
    try:
        data = [(u, f, float(o)) for u, f, o in data]
    except ValueError:
        data=[]
        print("Data could not be loaded.\nFound a value that is not a number.")
    except TypeError:
        data=[]
        print("Data could not be loaded.\nFound a row that is not a three element tuple.")
    except Exception as e:
        data=[]
        print(f"Data could not be loaded.\n{e}")

    return data
