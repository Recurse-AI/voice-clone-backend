from typing import List
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.pricing import Pricing  
from app.utils.logger import logger
from app.config.database import db

async def init_pricing_plans():
    collection = db["pricings"]

    existing_plans = await collection.count_documents({})
    if existing_plans > 0:
        logger.info("Pricing plans already exist, skipping initialization")
        return

    pricing_plans: List[dict] = [
        {
            "name": "small",
            "description": "Perfect for small projects",
            "features": [
                "150 credits",
                "All basic features",
                "Priority support",
                "Extended format support",
                "20% savings on credits"
            ],
            "creditPack": {
                "name": "Small",
                "credits": 150,
                "originalPrice": 6.00,
                "discountedPrice": 5.00,
                "pricePerCredit": 0.033,
                "savingsPercentage": 20,
                "stripePriceId": "price_small_pack"
            },
            "isActive": True,
            "displayOrder": 1,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        },
        {
            "name": "medium",
            "description": "Most popular choice for regular users",
            "features": [
                "350 credits",
                "All features included",
                "Priority support",
                "Batch processing",
                "40% savings on credits",
                "Extended file format support"
            ],
            "creditPack": {
                "name": "Medium",
                "credits": 350,
                "originalPrice": 14.00,
                "discountedPrice": 10.00,
                "pricePerCredit": 0.028,
                "savingsPercentage": 40,
                "stripePriceId": "price_medium_pack"
            },
            "isActive": True,
            "displayOrder": 2,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        },
        {
            "name": "special",
            "description": "Best value for regular users",
            "features": [
                "850 credits",
                "All premium features",
                "Priority support",
                "Advanced batch processing",
                "70% savings on credits",
                "Priority processing queue"
            ],
            "creditPack": {
                "name": "Special",
                "credits": 850,
                "originalPrice": 34.00,
                "discountedPrice": 20.00,
                "pricePerCredit": 0.023,
                "savingsPercentage": 70,
                "stripePriceId": "price_special_pack"
            },
            "isActive": True,
            "displayOrder": 3,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        },
        {
            "name": "limited",
            "description": "Best savings for power users",
            "features": [
                "1800 credits",
                "All premium features",
                "Priority support",
                "Advanced batch processing",
                "85% savings on credits",
                "Highest priority queue",
                "Dedicated account manager"
            ],
            "creditPack": {
                "name": "Limited",
                "credits": 1850,
                "originalPrice": 72.00,
                "discountedPrice": 40.00,
                "pricePerCredit": 0.02,
                "savingsPercentage": 85,
                "stripePriceId": "price_limited_pack"
            },
            "isActive": True,
            "displayOrder": 4,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        }
    ]

    for plan in pricing_plans:
        try:
            await collection.update_one(
                {"name": plan["name"]},
                {"$setOnInsert": plan},
                upsert=True
            )
            logger.info(f"Upserted pricing plan: {plan['name']}")
        except Exception as e:
            logger.error(f"Failed to upsert pricing plan {plan['name']}: {e}")

    logger.info("Successfully initialized pricing plans")