import os
from rq import Queue
from redis import Redis

def get_redis_connection():
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    return Redis.from_url(redis_url)

def get_dub_queue():
    redis_conn = get_redis_connection()
    return Queue("dub_queue", connection=redis_conn)

def get_separation_queue():
    redis_conn = get_redis_connection()
    return Queue("separation_queue", connection=redis_conn)
