from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from app.config.settings import settings  
from app.utils.logger import logger

# Global MongoDB client and database
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client[settings.DB_NAME]

# Collections
separation_jobs_collection: AsyncIOMotorCollection = db.separation_jobs
dub_jobs_collection: AsyncIOMotorCollection = db.dub_jobs
users_collection: AsyncIOMotorCollection = db.users
pricing_collection: AsyncIOMotorCollection = db.pricing

async def verify_connection():
    try:
        await client.admin.command("ping")
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.info(f"MongoDB connection failed: {e}")

# Export collections for easy import
__all__ = [
    "client", 
    "db", 
    "separation_jobs_collection", 
    "dub_jobs_collection", 
    "users_collection", 
    "pricing_collection",
    "verify_connection"
]
