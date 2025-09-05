"""
Base Repository - Clean database operations
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from app.config.settings import settings

logger = logging.getLogger(__name__)


class BaseRepository:
    """
    Base repository class for clean database operations
    Pure async operations with proper connection handling
    """
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Get MongoDB collection with lazy initialization"""
        if self._collection is None:
            self._client = AsyncIOMotorClient(settings.MONGODB_URI)
            db = self._client[settings.DB_NAME]
            self._collection = db[self.collection_name]
        return self._collection
    
    async def create(self, data: Dict[str, Any]) -> Optional[str]:
        """Create a new document"""
        try:
            data["created_at"] = datetime.now(timezone.utc)
            data["updated_at"] = datetime.now(timezone.utc)
            
            result = await self.collection.insert_one(data)
            logger.info(f"✅ Created document in {self.collection_name}: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create document in {self.collection_name}: {e}")
            return None
    
    async def get_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get document by job_id"""
        try:
            doc = await self.collection.find_one({"job_id": job_id})
            if doc:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
            return doc
            
        except Exception as e:
            logger.error(f"Failed to get document {job_id} from {self.collection_name}: {e}")
            return None
    
    async def update(self, job_id: str, update_data: Dict[str, Any]) -> bool:
        """Update document by job_id"""
        try:
            update_data["updated_at"] = datetime.now(timezone.utc)
            
            result = await self.collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated document in {self.collection_name}: {job_id}")
                return True
            else:
                logger.warning(f"❌ No update for {job_id} in {self.collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update {job_id} in {self.collection_name}: {e}")
            return False
    
    async def get_user_jobs(self, user_id: str, page: int = 1, limit: int = 50) -> Tuple[List[Dict[str, Any]], int]:
        """Get paginated jobs for a user"""
        try:
            query = {"user_id": user_id}
            
            # Get total count
            total_count = await self.collection.count_documents(query)
            
            # Get paginated results
            skip = (page - 1) * limit
            cursor = self.collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
            
            jobs = []
            async for doc in cursor:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                jobs.append(doc)
            
            return jobs, total_count
            
        except Exception as e:
            logger.error(f"Failed to get user jobs from {self.collection_name}: {e}")
            return [], 0
    
    async def delete(self, job_id: str, user_id: str) -> bool:
        """Delete document with user ownership check"""
        try:
            # Check ownership first
            doc = await self.get_by_id(job_id)
            if not doc or doc.get("user_id") != user_id:
                logger.warning(f"Delete denied for {job_id} in {self.collection_name}: ownership check failed")
                return False
            
            result = await self.collection.delete_one({"job_id": job_id})
            
            if result.deleted_count > 0:
                logger.info(f"✅ Deleted document from {self.collection_name}: {job_id}")
                return True
            else:
                logger.warning(f"❌ No delete for {job_id} in {self.collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete {job_id} from {self.collection_name}: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None
