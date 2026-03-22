import sqlite3

conn = sqlite3.connect("data/face_tracker.db")
cur = conn.cursor()

print("EVENTS COLUMNS:")
for row in cur.execute("PRAGMA table_info(events)"):
    print(row)

conn.close()
