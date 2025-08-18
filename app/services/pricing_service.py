from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from datetime import datetime
from app.config.database import db
from app.models.pricing import Pricing, CreditPack
from app.utils.logger import logger
from fastapi.encoders import jsonable_encoder
from bson import ObjectId

class PricingService:
    def __init__(self):
        self.collection = db["pricings"]

    async def get_all_plans(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get all active pricing plans ordered by displayOrder"""
        try:
            query = {} if include_inactive else {"isActive": True}
            cursor = self.collection.find(query).sort("displayOrder", 1)
            plans = await cursor.to_list(length=None)
            if not plans:
                logger.info(f"no plans found")
                return HTTPException(status_code=404, detail="No pricing plans found")
            return {
                "status_code": 200,
                "content": self.mongo_json({
                    "success": True,
                    "plans": plans,
                    "count": len(plans)
                })
            }
        except Exception as e:
            logger.error(f"Failed to get pricing plans: {e}")
            return []
        
    def mongo_json(self, data):
        return jsonable_encoder(
            data,
            custom_encoder={
                ObjectId: str,
                datetime: lambda v: v.isoformat()
            }
        )

    async def get_plan_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific pricing plan by name"""
        try:
            plan = await self.collection.find_one({"name": name, "isActive": True})
            return plan
        except Exception as e:
            logger.error(f"Failed to get pricing plan {name}: {e}")
            return None

    async def get_plan_by_id(self, plan_id: str) -> Dict:
        """Get a specific pricing plan by ID"""
        try:
            plan = await self.collection.find_one({"_id": ObjectId(plan_id), "isActive": True})
            return plan
        except Exception as e:
            logger.error(f"Failed to get pricing plan by ID {plan_id}: {e}")
            return None
        

    async def get_credit_pack_details(self, plan_id: str) -> dict:
        """Get credit pack details from a plan"""
        try:
            plan = await self.get_plan_by_id(plan_id)
            if not plan or "creditPack" not in plan:
                return None

            credit_pack = plan["creditPack"]
            return {
                "name": credit_pack["name"],
                "credits": credit_pack["credits"],
                "original_price": credit_pack["originalPrice"],
                "final_price": credit_pack["discountedPrice"],
                "discount_percentage": credit_pack["savingsPercentage"],
                "stripe_price_id": credit_pack["stripePriceId"],
                "price_per_credit": credit_pack["pricePerCredit"]
            }
        except Exception as e:
            logger.error(f"Failed to get credit pack details: {e}")
            return None
    async def get_credit_pack_details_by_name(self, plan_name: str) -> dict:
        """Get credit pack details from a plan"""
        try:
            plan = await self.get_plan_by_name(plan_name)
            if not plan or "creditPack" not in plan:
                return None

            credit_pack = plan["creditPack"]
            return {
                "name": credit_pack["name"],
                "credits": credit_pack["credits"],
                "original_price": credit_pack["originalPrice"],
                "final_price": credit_pack["discountedPrice"],
                "discount_percentage": credit_pack["savingsPercentage"],
                "stripe_price_id": credit_pack["stripePriceId"],
                "price_per_credit": credit_pack["pricePerCredit"]
            }
        except Exception as e:
            logger.error(f"Failed to get credit pack details: {e}")
            return None



    async def get_credit_pack_by_stripe_id(self, stripe_price_id: str) -> Optional[Dict[str, Any]]:
        """Get credit pack details by Stripe price ID"""
        try:
            plan = await self.collection.find_one(
                {"creditPack.stripePriceId": stripe_price_id, "isActive": True}
            )
            return plan.get("creditPack") if plan else None
        except Exception as e:
            logger.error(f"Failed to get credit pack by Stripe ID {stripe_price_id}: {e}")
            return None

    async def calculate_credits_price(self, credits: int) -> Dict[str, Any]:
        """Calculate the best price for a given number of credits"""
        try:
            # Get all active plans with credit packs
            plans = await self.get_all_active_plans()
            best_pack = None
            lowest_total_price = float('inf')

            for plan in plans:
                credit_pack = plan.get("creditPack")
                if not credit_pack:
                    continue

                # Calculate how many packs needed
                packs_needed = (credits + credit_pack["credits"] - 1) // credit_pack["credits"]
                total_price = packs_needed * credit_pack["discountedPrice"]

                if total_price < lowest_total_price:
                    lowest_total_price = total_price
                    best_pack = {
                        "name": credit_pack["name"],
                        "packs_needed": packs_needed,
                        "total_credits": packs_needed * credit_pack["credits"],
                        "price_per_credit": credit_pack["pricePerCredit"],
                        "total_price": total_price,
                        "savings_percentage": credit_pack["savingsPercentage"]
                    }

            return {
                "success": True,
                "credits_requested": credits,
                "best_pack": best_pack
            }
        except Exception as e:
            logger.error(f"Failed to calculate credits price: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_plan_status(self, plan_id: str, is_active: bool) -> bool:
        """Update the active status of a pricing plan"""
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(plan_id)},
                {
                    "$set": {
                        "isActive": is_active,
                        "updatedAt": datetime.now()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update plan status: {e}")
            return False

    async def update_stripe_price_id(self, plan_id: str, stripe_price_id: str) -> bool:
        """Update the Stripe price ID for a credit pack"""
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(plan_id)},
                {
                    "$set": {
                        "creditPack.stripePriceId": stripe_price_id,
                        "updatedAt": datetime.now()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update Stripe price ID: {e}")
            return False

    async def get_free_credits_amount(self) -> int:
        """Get the number of free credits given to new users"""
        try:
            free_plan = await self.get_plan_by_name("free")
            if free_plan and "features" in free_plan:
                for feature in free_plan["features"]:
                    if "free credits" in feature.lower():
                        import re
                        match = re.search(r'(\d+)\s+free credits', feature.lower())
                        if match:
                            return int(match.group(1))
            return 25  
        except Exception as e:
            logger.error(f"Failed to get free credits amount: {e}")
            return 25 

pricing_service = PricingService()
