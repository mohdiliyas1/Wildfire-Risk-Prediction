import sqlite3

conn = sqlite3.connect('wildfire_data.db')
cursor = conn.cursor()

# Add latitude column
try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN latitude REAL")
    print("✅ 'latitude' column added.")
except sqlite3.OperationalError:
    print("⚠️ 'latitude' column already exists.")

# Add longitude column
try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN longitude REAL")
    print("✅ 'longitude' column added.")
except sqlite3.OperationalError:
    print("⚠️ 'longitude' column already exists.")

conn.commit()
conn.close()
