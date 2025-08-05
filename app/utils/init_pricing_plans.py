from typing import List
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.pricing import Pricing  # adjust this import path as needed
from app.utils.logger import logger
from app.config.database import db

async def init_pricing_plans():
    collection = db.get_collection("pricings")

    existing_plans = await collection.count_documents({})
    if existing_plans > 0:
        logger.info("Pricing plans already exist, skipping initialization")
        return

    pricing_plans: List[dict] = [
        {
            "name": "registered",
            "description": "Ideal for regular users who need more processing time and better features",
            "features": [
                "25 minutes of total processing time",
                "High-quality audio output",
                "Extended file format support (MP3, WAV, FLAC, M4A)",
                "Priority email support",
                "No ads",
                "Batch processing (up to 3 files)",
                "Processing history and management"
            ],
            "isFree": True,
            "freeCreditGift": 0,
            "creditBased": False,
            "limits": {
                "maxDuration": 25,
                "maxFileSize": 250,
                "maxConcurrentJobs": 2,
                "requiresCaptcha": False,
                "showsAds": False,
                "priority": "medium",
                "batchProcessing": True
            },
            "outputFormats": ["mp3", "wav", "flac", "m4a"],
            "isActive": True,
            "displayOrder": 2,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        },
        {
            "name": "starter",
            "description": "Perfect for users who need more than free but not full premium",
            "features": [
                "500 credits (≈ 8+ hours processing)",
                "Premium audio quality",
                "Priority processing queue",
                "Credits never expire",
                "All file formats supported",
                "Email support"
            ],
            "isFree": False,
            "creditBased": True,
            "creditsPerMinute": 1,
            "pricing": {
                "basePackage": {
                    "price": 20.00,
                    "credits": 500
                }
            },
            "stripePriceIds": {
                "basePackage": "price_starter_20_credits"
            },
            "limits": {
                "maxFileSize": 500,
                "maxConcurrentJobs": 3,
                "requiresCaptcha": False,
                "showsAds": False,
                "priority": "high",
                "batchProcessing": True,
                "fullApiAccess": True
            },
            "outputFormats": ["mp3", "wav", "flac", "m4a", "aac"],
            "isActive": True,
            "displayOrder": 4,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        },
        {
            "name": "premium",
            "description": "Best for professionals and heavy users with unlimited access and premium features",
            "features": [
                "Pay-per-use credit system (1 credit = 1 minute)",
                "Premium audio quality with AI enhancement",
                "All file formats supported",
                "Priority processing queue",
                "24/7 priority support",
                "Advanced AI models and separation algorithms",
                "API access for automation",
                "Custom output formats and settings",
                "Credits never expire",
                "Bulk processing discounts"
            ],
            "isFree": False,
            "creditBased": True,
            "creditsPerMinute": 1,
            "pricing": {
                "creditPacks": [
                    {"credits": 25, "price": 1.00, "discountPercentage": 0},
                    {"credits": 250, "price": 9.00, "discountPercentage": 10},
                    {"credits": 500, "price": 20.00, "discountPercentage": 0},
                    {"credits": 625, "price": 20.00, "discountPercentage": 20},
                    {"credits": 1250, "price": 37.50, "discountPercentage": 25},
                    {"credits": 2500, "price": 70.00, "discountPercentage": 30},
                    {"credits": 5000, "price": 130.00, "discountPercentage": 35}
                ],
                "basePackage": {
                    "price": 1.00,
                    "credits": 25
                },
                "customCredits": {
                    "minPrice": 100.00,
                    "minCredits": 2500,
                    "pricePerCredit": 0.04
                }
            },
            "stripePriceIds": {
                "creditPack25": "price_premium_25_credits",
                "creditPack250": "price_premium_250_credits",
                "creditPack500": "price_premium_500_credits",
                "creditPack625": "price_premium_625_credits",
                "creditPack1250": "price_premium_1250_credits",
                "creditPack2500": "price_premium_2500_credits",
                "creditPack5000": "price_premium_5000_credits",
                "customCredits": "price_premium_custom_credits"
            },
            "limits": {
                "maxFileSize": 1000,
                "maxConcurrentJobs": 10,
                "requiresCaptcha": False,
                "showsAds": False,
                "priority": "high",
                "hasLimitedApiAccess": False,
                "batchProcessing": True,
                "fullApiAccess": True
            },
            "outputFormats": ["mp3", "wav", "flac", "m4a", "aiff", "ogg"],
            "isActive": True,
            "displayOrder": 3,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
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

