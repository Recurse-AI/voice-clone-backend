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
            "name": "free",
            "description": "Start with free credits and explore our features",
            "features": [
                "25 free credits upon signup",
                "Basic audio processing",
                "Standard support",
                "Common file formats supported",
                "Standard processing queue"
            ],
            "creditPack": None,  
            "isActive": True,
            "displayOrder": 0,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        },
        {
            "name": "small",
            "description": "Perfect for small projects",
            "features": [
                "75 credits",
                "All basic features",
                "Priority support",
                "Extended format support",
                "17% savings on credits"
            ],
            "creditPack": {
                "name": "Small",
                "credits": 75,
                "originalPrice": 6.00,
                "discountedPrice": 4.99,
                "pricePerCredit": 0.0665,
                "savingsPercentage": 17,
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
                "400 credits",
                "All features included",
                "Priority support",
                "Batch processing",
                "38% savings on credits",
                "Extended file format support"
            ],
            "creditPack": {
                "name": "Medium",
                "credits": 400,
                "originalPrice": 16.00,
                "discountedPrice": 9.99,
                "pricePerCredit": 0.0249,
                "savingsPercentage": 38,
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
                "1000 credits",
                "All premium features",
                "Priority support",
                "Advanced batch processing",
                "50% savings on credits",
                "Premium file format support",
                "Priority processing queue"
            ],
            "creditPack": {
                "name": "Special",
                "credits": 1000,
                "originalPrice": 40.00,
                "discountedPrice": 19.99,
                "pricePerCredit": 0.0199,
                "savingsPercentage": 50,
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
                "2500 credits",
                "All premium features",
                "Priority support",
                "Advanced batch processing",
                "60% savings on credits",
                "Premium file format support",
                "Highest priority queue",
                "Dedicated account manager"
            ],
            "creditPack": {
                "name": "Limited",
                "credits": 2500,
                "originalPrice": 100.00,
                "discountedPrice": 39.99,
                "pricePerCredit": 0.0159,
                "savingsPercentage": 60,
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