"""
Database Index Initialization
Ensures unique indexes on job_id fields to prevent duplicate processing across multiple workers
"""

import logging
from typing import List, Dict, Any
from app.config.database import dub_jobs_collection, separation_jobs_collection

logger = logging.getLogger(__name__)


async def init_database_indexes():
    """
    Initialize database indexes for production-safe duplicate prevention.
    Creates unique indexes on job_id fields to ensure no duplicate processing
    even with multiple workers.
    """
    indexes_to_create = [
        {
            "collection": dub_jobs_collection,
            "name": "dub_jobs.job_id",
            "field": "job_id"
        },
        {
            "collection": separation_jobs_collection,
            "name": "separation_jobs.job_id", 
            "field": "job_id"
        },


    ]
    # Non-unique performance indexes
    non_unique_indexes_to_create = [
        # Optimize user list pagination: filter by user_id, sort by created_at
        {
            "collection": dub_jobs_collection,
            "name": "dub_jobs.user_id_created_at",
            "fields": [("user_id", 1), ("created_at", -1)]
        },
        {
            "collection": separation_jobs_collection,
            "name": "separation_jobs.user_id_created_at",
            "fields": [("user_id", 1), ("created_at", -1)]
        },
        # Optimize statistics counts: filter by user_id and status
        {
            "collection": dub_jobs_collection,
            "name": "dub_jobs.user_id_status",
            "fields": [("user_id", 1), ("status", 1)]
        },
        {
            "collection": separation_jobs_collection,
            "name": "separation_jobs.user_id_status",
            "fields": [("user_id", 1), ("status", 1)]
        },
    ]
    
    # TTL indexes for automatic cleanup
    ttl_indexes_to_create = []
    
    created_count = 0
    existing_count = 0
    
    try:
        # Create unique indexes
        for index_config in indexes_to_create:
            try:
                await index_config["collection"].create_index(
                    index_config["field"], 
                    unique=True,
                    background=True,  # Non-blocking index creation
                    sparse=True  # Skip null values for better performance
                )
                logger.info(f"Created unique index on {index_config['name']}")
                created_count += 1
                
            except Exception as index_error:
                if any(keyword in str(index_error).lower() for keyword in 
                       ["already exists", "index already exists", "duplicate key"]):
                    logger.info(f"📋 Index on {index_config['name']} already exists")
                    existing_count += 1
                else:
                    logger.warning(f"⚠️ Failed to create index on {index_config['name']}: {index_error}")
        # Create non-unique performance indexes
        for idx in non_unique_indexes_to_create:
            try:
                await idx["collection"].create_index(
                    idx.get("fields") or idx.get("field"),
                    background=True
                )
                logger.info(f"Created index on {idx['name']}")
                created_count += 1
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in 
                       ["already exists", "index already exists"]):
                    logger.info(f"📋 Index on {idx['name']} already exists")
                    existing_count += 1
                else:
                    logger.warning(f"⚠️ Failed to create index on {idx['name']}: {e}")
        
        # Create TTL indexes for automatic cleanup
        for ttl_config in ttl_indexes_to_create:
            try:
                await ttl_config["collection"].create_index(
                    ttl_config["field"],
                    expireAfterSeconds=ttl_config["expire_after_seconds"],
                    background=True
                )
                logger.info(f"🗑️ Created TTL index on {ttl_config['name']} (expires after {ttl_config['expire_after_seconds']}s)")
                created_count += 1
                
            except Exception as ttl_error:
                if any(keyword in str(ttl_error).lower() for keyword in 
                       ["already exists", "index already exists"]):
                    logger.info(f"📋 TTL index on {ttl_config['name']} already exists")
                    existing_count += 1
                else:
                    logger.warning(f"⚠️ Failed to create TTL index on {ttl_config['name']}: {ttl_error}")
        
        logger.info(f"🎯 Database indexes summary: {created_count} created, {existing_count} existing")
        
    except Exception as e:
        logger.error(f"❌ Critical error during index initialization: {e}")
        # Don't raise exception - app should continue even if index creation fails
