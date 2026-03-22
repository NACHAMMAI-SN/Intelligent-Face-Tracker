import sqlite3

conn = sqlite3.connect("data/face_tracker.db")
cur = conn.cursor()

print("persons =", cur.execute("select count(*) from persons").fetchone()[0])
print("registered_events =", cur.execute("select count(*) from events where event_type='REGISTERED'").fetchone()[0])
print("entries =", cur.execute("select count(*) from events where event_type='ENTRY'").fetchone()[0])
print("exits =", cur.execute("select count(*) from events where event_type='EXIT'").fetchone()[0])

print("\nSample events:")
for row in cur.execute("select event_type, person_id, event_time from events limit 10"):
    print(row)

conn.close()
