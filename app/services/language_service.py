"""
Centralized Language Service
Handles all language validation, mapping, and normalization
"""

from typing import Set, Dict
import logging

logger = logging.getLogger(__name__)

class LanguageService:
    """
    Centralized service for language handling across the application.
    Single source of truth for all language operations.
    """
    
    # Language mapping - includes all supported languages (source + target)
    LANGUAGE_NAME_TO_CODE: Dict[str, str] = {
        # Core supported languages
        "english": "en",
        "chinese": "zh", 
        "dutch": "nl",
        "finnish": "fi",  # WhisperX transcription only
        "french": "fr",
        "german": "de",
        "hindi": "hi",
        "italian": "it",
        "japanese": "ja",
        "korean": "ko",
        "polish": "pl",
        "portuguese": "pt",
        "russian": "ru",
        "spanish": "es",
        "turkish": "tr",
        "ukrainian": "uk",
        "vietnamese": "vi",
        "arabic": "ar",  # Fish Speech dubbing only
    }
    
    # Languages supported for dubbing (Fish Speech) - matches frontend target languages exactly
    DUBBING_SUPPORTED_CODES: Set[str] = {
        "en", "zh", "ja", "de", "fr", "es", "ko", "ar", "ru", 
        "tr", "uk", "vi", "hi", "nl", "it", "pl", "pt"
    }
    
    # Languages supported for transcription (WhisperX) - matches frontend exactly
    TRANSCRIPTION_SUPPORTED_CODES: Set[str] = {
        "en", "zh", "nl", "fi", "fr", "de", "hi", "it", "ja", 
        "ko", "pl", "pt", "ru", "es", "tr", "uk", "vi"
    }
    
    # Accepted tokens that mean: let the system auto-detect source language
    AUTO_DETECT_TOKENS: Set[str] = {"auto", "auto_detect", "auto-detect", "auto detect"}
    
    @classmethod
    def normalize_language_input(cls, language: str) -> str:
        """
        Normalize language input to supported format.
        Converts language names to codes and validates support.
        """
        if not language:
            return "en"  # Default to English
        
        language_lower = language.lower().strip()
        
        # Handle auto-detect sentinel values
        if language_lower in cls.AUTO_DETECT_TOKENS:
            return "auto_detect"
        
        # Direct mapping from name to code
        if language_lower in cls.LANGUAGE_NAME_TO_CODE:
            return cls.LANGUAGE_NAME_TO_CODE[language_lower]
        
        # If it's already a language code, validate and return
        if language_lower in cls.DUBBING_SUPPORTED_CODES or language_lower in cls.TRANSCRIPTION_SUPPORTED_CODES:
            return language_lower
        
        # Handle compound codes (en-US -> en) - but only if base code is supported
        if "-" in language_lower or "_" in language_lower:
            base_code = language_lower.split("-")[0].split("_")[0]
            if base_code in cls.DUBBING_SUPPORTED_CODES or base_code in cls.TRANSCRIPTION_SUPPORTED_CODES:
                return base_code
        
        # Return default if not found
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
        """Check if a language is supported for transcription (WhisperX)."""
        if not language:
            return False
        normalized = cls.normalize_language_input(language)
        # Allow auto-detect as a valid option for transcription
        if normalized == "auto_detect":
            return True
        return normalized in cls.TRANSCRIPTION_SUPPORTED_CODES
    
    @classmethod
    def get_supported_dubbing_languages(cls) -> Set[str]:
        """Get all supported language names for dubbing."""
        return {name for name, code in cls.LANGUAGE_NAME_TO_CODE.items() 
                if code in cls.DUBBING_SUPPORTED_CODES}
    
    @classmethod
    def get_language_code_for_transcription(cls, language: str) -> str:
        """
        Get language code for transcription.
        Accepts 'auto_detect' to enable automatic language detection.
        """
        if not language:
            raise ValueError("Language is required for transcription (use 'auto_detect' to auto-detect)")
        
        normalized = cls.normalize_language_input(language)
        
        # Allow auto-detect to pass through
        if normalized == "auto_detect":
            return normalized
        
        if normalized not in cls.TRANSCRIPTION_SUPPORTED_CODES:
            raise ValueError(f"Language '{language}' not supported for transcription")
        
        return normalized


# Global service instance
language_service = LanguageService()
