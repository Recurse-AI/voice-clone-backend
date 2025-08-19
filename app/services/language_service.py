"""
Centralized Language Service
Handles all language validation, mapping, and normalization
"""

from typing import Optional, Set, Dict
import logging

logger = logging.getLogger(__name__)

class LanguageService:
    """
    Centralized service for language handling across the application.
    Single source of truth for all language operations.
    """
    
    # Complete language mapping with all supported languages
    LANGUAGE_NAME_TO_CODE: Dict[str, Optional[str]] = {
        # Special cases
        "auto detect": None,
        
        # Core supported languages (Fish Speech + Assembly AI)
        "english": "en",
        "english_global": "en", 
        "english_us": "en-US",
        "english_au": "en-AU", 
        "english_uk": "en-GB",
        "chinese": "zh",
        "japanese": "ja",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "korean": "ko",
        "arabic": "ar",
        "russian": "ru",
        "dutch": "nl",
        "italian": "it",
        "polish": "pl",
        "portuguese": "pt",
        "hindi": "hi",
        "turkish": "tr",
        "ukrainian": "uk",
        "vietnamese": "vi",
        
        # Additional Assembly AI supported languages
        "azerbaijani": "az",
        "czech": "cs",
        "danish": "da",
        "finnish": "fi",
        "hebrew": "he",
        "hungarian": "hu",
        "indonesian": "id",
        "norwegian": "no",
        "romanian": "ro",
        "swedish": "sv",
    }
    
    # Languages supported for dubbing (Fish Speech)
    DUBBING_SUPPORTED_CODES: Set[str] = {
        "en", "zh", "ja", "de", "fr", "es", "ko", "ar", 
        "ru", "nl", "it", "pl", "pt", "hi", "tr", "uk", "vi"
    }
    
    # Languages supported for transcription (Assembly AI)
    TRANSCRIPTION_SUPPORTED_CODES: Set[str] = {
        "en", "en-US", "en-AU", "en-GB", "zh", "ja", "de", "fr", "es", 
        "ko", "ar", "ru", "nl", "it", "pl", "pt", "hi", "tr", "uk", "vi",
        "az", "cs", "da", "fi", "he", "hu", "id", "no", "ro", "sv"
    }
    
    @classmethod
    def normalize_language_input(cls, language: str) -> str:
        """
        Normalize language input to supported format.
        Converts language names to codes and validates support.
        """
        if not language:
            return "en"  # Default to English
        
        language_lower = language.lower().strip()
        
        # Direct mapping from name to code
        if language_lower in cls.LANGUAGE_NAME_TO_CODE:
            code = cls.LANGUAGE_NAME_TO_CODE[language_lower]
            return code if code else "en"  # Handle None (auto detect) -> default English
        
        # If it's already a language code, validate and return
        if language_lower in cls.DUBBING_SUPPORTED_CODES or language_lower in cls.TRANSCRIPTION_SUPPORTED_CODES:
            return language_lower
        
        # Handle compound codes (en-US -> en)
        if "-" in language_lower or "_" in language_lower:
            base_code = language_lower.split("-")[0].split("_")[0]
            if base_code in cls.DUBBING_SUPPORTED_CODES or base_code in cls.TRANSCRIPTION_SUPPORTED_CODES:
                return base_code
        
        # Return original if not found (will be caught by validation)
        logger.warning(f"Unknown language input: {language}, defaulting to English")
        return "en"
    
    @classmethod
    def is_dubbing_supported(cls, language: str) -> bool:
        """Check if a language is supported for dubbing (Fish Speech)."""
        if not language:
            return False
        normalized = cls.normalize_language_input(language)
        return normalized in cls.DUBBING_SUPPORTED_CODES
    
    @classmethod 
    def is_transcription_supported(cls, language: str) -> bool:
        """Check if a language is supported for transcription (Assembly AI)."""
        if not language:
            return True  # Auto detect supported
        normalized = cls.normalize_language_input(language)
        return normalized in cls.TRANSCRIPTION_SUPPORTED_CODES
    
    @classmethod
    def get_supported_dubbing_languages(cls) -> Set[str]:
        """Get all supported language names for dubbing."""
        return {name for name, code in cls.LANGUAGE_NAME_TO_CODE.items() 
                if code and code in cls.DUBBING_SUPPORTED_CODES}
    
    @classmethod
    def get_language_code_for_transcription(cls, language: Optional[str]) -> Optional[str]:
        """
        Get language code suitable for Assembly AI transcription.
        Returns None for auto-detect.
        """
        if not language or language.lower().strip() == "auto detect":
            return None
        
        normalized = cls.normalize_language_input(language)
        
        # For transcription, we can use more specific codes if available
        language_lower = language.lower().strip()
        if language_lower in cls.LANGUAGE_NAME_TO_CODE:
            return cls.LANGUAGE_NAME_TO_CODE[language_lower]
        
        return normalized if normalized in cls.TRANSCRIPTION_SUPPORTED_CODES else None


# Global service instance
language_service = LanguageService()
