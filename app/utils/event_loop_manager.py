"""
Event loop manager to handle async operations from sync contexts.
Stores reference to main event loop for use in background threads.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class EventLoopManager:
    """Manages reference to the main event loop for background thread usage"""
    
    def __init__(self):
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
    
    def set_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store reference to the main event loop"""
        self._main_loop = loop
        logger.info(f"✅ Main event loop registered: {loop}")
    
    def get_main_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the stored main event loop"""
        return self._main_loop
    
    def is_main_loop_available(self) -> bool:
        """Check if main loop is available and running"""
        return self._main_loop is not None and self._main_loop.is_running()

# Global instance
loop_manager = EventLoopManager()
