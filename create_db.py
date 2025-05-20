import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS explosions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value REAL
)
""")
conn.commit()
conn.close()

print("تم إنشاء قاعدة البيانات بنجاح ✅")
