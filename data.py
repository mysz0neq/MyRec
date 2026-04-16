"""Introduces data to the pipeline"""

import sqlite3
from pathlib import Path

PATH=Path(__file__).parent
DB_PATH=PATH.joinpath("baza.db")

def get_data(path:Path=DB_PATH) -> list[tuple[str,str,float]]:
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM oceny''')
    data = cursor.fetchall()
    try:
        data=[(u,f,float(o)) for u,f,o in data]
    except:
        print("Data could not be loaded")
    return data

