#!/usr/bin/env python3
"""
Simple Billing Cron - $10 chunks billing every 3 hours
Usage: python -m app.utils.billing_cron
"""

import asyncio
import logging
from datetime import datetime
from app.services.background_billing_service import background_billing_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_billing():
    """Run billing for all PAYG users"""
    try:
        logger.info(f"Starting billing at {datetime.now()}")
        results = await background_billing_service.process_all_payg_users()
        
        if "error" in results:
            logger.error(f"Billing failed: {results['error']}")
        else:
            logger.info(f"Billing done: {results['billed']} billed, {results['blocked']} blocked")
            
    except Exception as e:
        logger.error(f"Billing error: {e}")

if __name__ == "__main__":
    asyncio.run(run_billing())
