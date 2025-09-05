from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import MongoClient
from app.config.settings import settings  
import logging
from typing import Optional
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Global MongoDB clients and database
client = AsyncIOMotorClient(settings.MONGODB_URI)
sync_client = MongoClient(settings.MONGODB_URI)
db = client[settings.DB_NAME]

# Collections
separation_jobs_collection: AsyncIOMotorCollection = db.separation_jobs
dub_jobs_collection: AsyncIOMotorCollection = db.dub_jobs
users_collection: AsyncIOMotorCollection = db.users
pricing_collection: AsyncIOMotorCollection = db.pricing
transaction_collection: AsyncIOMotorCollection = db.creditTransaction



_loop_local_async_client: ContextVar[Optional[AsyncIOMotorClient]] = ContextVar(
    "_loop_local_async_client", default=None
)

def get_async_db():
    """Return an AsyncIOMotor database bound to the current event loop.

    Creates a loop-local AsyncIOMotorClient on first use per asyncio context
    to avoid "Event loop is closed" issues when using asyncio.run in workers.
    """
    local_client = _loop_local_async_client.get()
    if local_client is None:
        local_client = AsyncIOMotorClient(settings.MONGODB_URI)
        _loop_local_async_client.set(local_client)
    return local_client[settings.DB_NAME]

async def verify_connection():
    try:
        await client.admin.command("ping")
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.info(f"MongoDB connection failed: {e}")

async def create_unique_indexes():
    """Create unique indexes for collections"""
    try:
        await db.creditTransaction.create_index("stripeSessionId", unique=True)
        logger.info("Created unique index for stripeSessionId in creditTransaction collection")
        await db.users.create_index("email", unique=True)
        
    except Exception as e:
        logger.warning(f"Failed to create unique indexes: {e}")


# Export collections for easy import
__all__ = [
    "client", 
    "sync_client",
    "db", 
    "separation_jobs_collection", 
    "dub_jobs_collection", 
    "users_collection", 
    "pricing_collection",
    "transaction_collection",
    "verify_connection"
]
