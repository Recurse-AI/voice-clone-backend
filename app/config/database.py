from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from app.config.settings import settings  
import logging

logger = logging.getLogger(__name__)

# Global MongoDB client and database
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client[settings.DB_NAME]

# Collections
separation_jobs_collection: AsyncIOMotorCollection = db.separation_jobs
dub_jobs_collection: AsyncIOMotorCollection = db.dub_jobs
users_collection: AsyncIOMotorCollection = db.users
pricing_collection: AsyncIOMotorCollection = db.pricing
transaction_collection: AsyncIOMotorCollection = db.creditTransaction



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
    "db", 
    "separation_jobs_collection", 
    "dub_jobs_collection", 
    "users_collection", 
    "pricing_collection",
    "transaction_collection",


    "verify_connection"
]
