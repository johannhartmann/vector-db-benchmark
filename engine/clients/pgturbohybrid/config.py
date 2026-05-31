import os

PGTURBOHYBRID_PORT = int(os.getenv("PGTURBOHYBRID_PORT", 5432))
PGTURBOHYBRID_DB = os.getenv("PGTURBOHYBRID_DB", "pgturbohybrid")
PGTURBOHYBRID_USER = os.getenv("PGTURBOHYBRID_USER", "postgres")
PGTURBOHYBRID_PASSWORD = os.getenv("PGTURBOHYBRID_PASSWORD", "postgres")


def get_db_config(host, connection_params):
    return {
        "host": host or "localhost",
        "port": PGTURBOHYBRID_PORT,
        "dbname": PGTURBOHYBRID_DB,
        "user": PGTURBOHYBRID_USER,
        "password": PGTURBOHYBRID_PASSWORD,
        "autocommit": True,
        **connection_params,
    }
