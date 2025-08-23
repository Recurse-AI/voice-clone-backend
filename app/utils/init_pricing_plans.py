from typing import List
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.pricing import Pricing  
import logging
from app.config.database import db

logger = logging.getLogger(__name__)

async def init_pricing_plans():
    collection = db["pricings"]

    existing_plans = await collection.count_documents({})
    if existing_plans > 0:
        logger.info("Pricing plans already exist, skipping initialization")
        return

    pricing_plans: List[dict] = [
        {
            "name": "pay-as-you-go",
            "description": "Perfect for flexible usage and occasional projects",
            "features": [
                "Flexible pricing for occasional use",
                "All basic features",
                "Priority support",
                "Extended format support",
                "Priority processing queue",
                "No upfront commitment",
                "Pay only for what you use"
            ],
            "creditPack": {
                "name": "Pay As You Go",
                "credits": 0,  # No upfront credits
                "originalPrice": 0.00,  # No upfront cost
                "discountedPrice": 0.00,  # No upfront cost
                "pricePerCredit": 0.04,  # $0.04 per credit (25 credits = $1.00)
                "savingsPercentage": 0,  # No discount for pay-as-you-go
                "stripePriceId": "price_pay_as_you_go"  # Your metered price ID
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
                "discountedPrice": 9.99,
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
                "discountedPrice": 19.99,
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
                "discountedPrice": 39.99,
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