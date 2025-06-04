import psycopg2
import uuid


class MetricsDB:
    def __init__(self, dbname, user, password, host="localhost", port=5432):
        self.connection = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
        self.cursor = self.connection.cursor()

    def insert_entry(self, data, data_type):
        try:
            data["id"] = str(uuid.uuid4())
            data["type"] = data_type
            data["population"] = 1000
            data["nodes"] = 1
            data["visibility"] = "all"

            self.cursor.execute(
                """
                INSERT INTO metrics (id, type, accuracy, anonymity, mia, population, nodes, visibility)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """,
                (
                    data["id"],  # ← ora è una stringa UUID valida
                    data["type"],
                    data["ML Accuracy"],
                    data["k-Anonymity"],
                    data["MIA Accuracy"],
                    data["population"],
                    data["nodes"],
                    data["visibility"],
                ),
            )
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise RuntimeError(f"Failed to insert metrics: {e}")

    def fetch_all(self):
        self.cursor.execute("SELECT * FROM metrics")
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.connection.close()
