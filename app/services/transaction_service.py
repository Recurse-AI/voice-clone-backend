from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from pymongo.errors import DuplicateKeyError
from app.config.database import db, transaction_collection
from app.utils.logger import logger
from fastapi.encoders import jsonable_encoder

class TransactionService:
    def __init__(self):
        self.collection = transaction_collection
        self.users_collection = db["users"]

    def serialize_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize transaction data for JSON response"""
        if not transaction:
            return None
            
        # Convert ObjectId and datetime objects
        serialized = jsonable_encoder(transaction)
        
        # Additional formatting if needed
        if "createdAt" in serialized:
            serialized["createdAt"] = serialized["createdAt"].isoformat() if isinstance(serialized["createdAt"], datetime) else serialized["createdAt"]
            
        return serialized

    async def get_transaction(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get existing transaction by session ID"""
        try:
            transaction = await self.collection.find_one({
                "stripeSessionId": session_id,
                "userId": user_id
            })
            return self.serialize_transaction(transaction)
        except Exception as e:
            logger.error(f"Error getting transaction: {e}")
            return None

    async def create_transaction(self, 
        user_id: str,
        credits: float,
        amount: float,
        session_id: str,
        description: str = None
    ) -> Dict[str, Any]:
        """Create a new credit transaction"""
        try:
            # Check for existing transaction
            existing = await self.get_transaction(session_id, user_id)
            if existing:
                return existing

            # Create new transaction
            transaction_data = {
                "userId": user_id,
                "type": "purchase",
                "credits": credits,
                "amount": amount,
                "stripeSessionId": session_id,
                "description": description or f"Purchase of {credits} credits",
                "status": "success",
                "createdAt": datetime.now(timezone.utc)
            }

            try:
                result = await self.collection.insert_one(transaction_data)
                transaction_data["_id"] = str(result.inserted_id)
                return self.serialize_transaction(transaction_data)

            except DuplicateKeyError:
                # If concurrent request created the transaction, get and return it
                raise HTTPException(status_code=400, detail="Transaction creation failed due to duplicate entry")

        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            raise HTTPException(status_code=500, detail="Failed to create transaction")

    async def update_user_credits(self, user_id: str, credits: float) -> Dict[str, Any]:
        """Update user's credits and subscription status"""
        try:
            update_time = datetime.now(timezone.utc)
            result = await self.users_collection.find_one_and_update(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {"credits": credits},
                    "$set": {
                        "subscription.type": "premium",
                        "subscription.status": "active",
                        "updatedAt": update_time
                    }
                },
                return_document=True
            )

            if not result:
                raise HTTPException(status_code=404, detail="User not found")

            return {
                "credits": result.get("credits", 0),
                "subscription": {
                    "type": "premium",
                    "status": "active"
                },
                "updatedAt": update_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Error updating user credits: {e}")
            raise HTTPException(status_code=500, detail="Failed to update user credits")

    async def get_purchase_history(self, user_id: str) -> Dict[str, Any]:
        """Get user's purchase history"""
        try:
            if self.collection is None:
                logger.info(f"---> collection is None")
                raise HTTPException(
                    status_code=500,
                    detail="Transaction service not initialized"
                )

            cursor = self.collection.find(
                {"userId": user_id}
            ).sort("createdAt", -1)
            
            # Get all transactions and convert each one
            transactions = []
            async for txn in cursor:
                # Convert ObjectId to string and format datetime
                formatted_txn = {
                    "_id": str(txn["_id"]),
                    "userId": txn["userId"],
                    "type": txn.get("type"),
                    "credits": txn.get("credits"),
                    "amount": txn.get("amount"),
                    "stripeSessionId": txn.get("stripeSessionId"),
                    "description": txn.get("description"),
                    "status": txn.get("status"),
                    "createdAt": txn["createdAt"].isoformat() if txn.get("createdAt") else None
                }
                transactions.append(formatted_txn)

            return {
                "status_code": 200,
                "content": {
                    "success": True,
                    "message": "Purchase history retrieved successfully",
                    "transactions": transactions
                }
            }
        except Exception as error:
            logger.error(f"Error retrieving purchase history: {str(error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve purchase history: {str(error)}"
            )

    async def handle_checkout_completed(self, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle successful checkout completion and create transaction"""
        try:
            # Check for existing transaction
            existing = await self.get_transaction(
                session_id=session["id"],
                user_id=session['metadata'].get('userId')
            )
            
            if existing:
                logger.info(f"Transaction already recorded for session: {session['id']}")
                return {
                    "success": True,
                    "message": "Payment already processed",
                    "transaction": existing
                }
            
            # Validate metadata
            user_id = session['metadata'].get('userId')
            credits = float(session['metadata'].get('credit', 0))
            
            if not user_id or not credits or credits <= 0:
                logger.error(f"Invalid metadata in session: {session['id']}")
                return None

            logger.info(f"Processing payment for {credits} credits for user: {user_id}")

            # Create transaction
            transaction = await self.create_transaction(
                user_id=user_id,
                credits=credits,
                amount=session["amount_total"] / 100,
                session_id=session["id"],
                description=f"Purchase of {credits} credits"
            )

            # Update user credits
            user_update = await self.update_user_credits(
                user_id=user_id,
                credits=credits
            )

            logger.info(f"Successfully processed payment for user {user_id}")
            
            return {
                "success": True,
                "message": "Payment processed successfully",
                "transaction": transaction,
                "user": user_update
            }

        except Exception as error:
            logger.error(f"Error handling checkout completed: {str(error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process checkout: {str(error)}"
            )

transaction_service = TransactionService()
