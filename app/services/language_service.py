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
        "english": "en",
        "chinese": "zh",
        "japanese": "ja",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "korean": "ko",
        "arabic": "ar",
        "russian": "ru",
        "turkish": "tr",
        "ukrainian": "uk",
        "vietnamese": "vi",
        "hindi": "hi",
        "dutch": "nl",
        "italian": "it",
        "polish": "pl",
        "portuguese": "pt",
        "bulgarian": "bg",
        "czech": "cs",
        "danish": "da",
        "greek": "el",
        "estonian": "et",
        "finnish": "fi",
        "hebrew": "he",
        "croatian": "hr",
        "hungarian": "hu",
        "indonesian": "id",
        "lithuanian": "lt",
        "latvian": "lv",
        "malay": "ms",
        "norwegian": "no",
        "romanian": "ro",
        "slovak": "sk",
        "slovenian": "sl",
        "serbian": "sr",
        "swedish": "sv",
        "thai": "th",
        "filipino": "fil",
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "urdu": "ur",
        "persian": "fa",
        "georgian": "ka",
        "armenian": "hy",
        "azeri": "az",
        "azerbaijani": "az",
        "kazakh": "kk",
        "nepali": "ne",
        "marathi": "mr",
        "punjabi": "pa",
        "gujarati": "gu",
        "kannada": "kn",
        "malayalam": "ml",
        "afrikaans": "af",
        "welsh": "cy",
        "icelandic": "is",
        "irish": "ga",
        "macedonian": "mk",
        "bosnian": "bs",
        "galician": "gl",
        "catalan": "ca",
        "swahili": "sw",
        "somali": "so",
        "hausa": "ha",
    }
    
    # Languages supported for dubbing (Fish Speech) - matches frontend target languages exactly
    DUBBING_SUPPORTED_CODES: Set[str] = {
        "en", "zh", "ja", "de", "fr", "es", "ko", "ar", "ru", 
        "tr", "uk", "vi", "hi", "nl", "it", "pl", "pt"
    }
    
    # Languages supported for transcription (WhisperX + AssemblyAI) - only languages supported by BOTH
    TRANSCRIPTION_SUPPORTED_CODES: Set[str] = {
        "en",  # English
        "es",  # Spanish
        "fr",  # French
        "de",  # German
        "it",  # Italian
        "pt",  # Portuguese
        "nl",  # Dutch
        "ru",  # Russian
        "pl",  # Polish
        "tr",  # Turkish
        "uk",  # Ukrainian
        "ar",  # Arabic
        "hi",  # Hindi
        "ja",  # Japanese
        "zh",  # Chinese (Mandarin)
        "ko",  # Korean
        "id",  # Indonesian
        "sv",  # Swedish
        "cs",  # Czech
        "fi",  # Finnish
        "el",  # Greek
        "da",  # Danish
        "hu",  # Hungarian
        "ro",  # Romanian
        "vi",  # Vietnamese
        "bg",  # Bulgarian
        "hr",  # Croatian
        "sk",  # Slovak
        "sl",  # Slovenian
        "sr"   # Serbian
    }

    
    # ElevenLabs V3 supported languages (70+ languages)
    ELEVENLABS_V3_LANGUAGES: Set[str] = {
        'af', 'ar', 'hy', 'as', 'az', 'be', 'bn', 'bs', 'bg', 'ca',
        'hr', 'cs', 'da', 'nl', 'en', 'et', 'fil', 'fi', 'fr', 'gl',
        'ka', 'de', 'el', 'gu', 'ha', 'he', 'hi', 'hu', 'is', 'id',
        'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'ky', 'ko', 'lv', 'lt',
        'lb', 'mk', 'ms', 'ml', 'zh', 'mr', 'ne', 'no', 'ps', 'fa',
        'pl', 'pt', 'pa', 'ro', 'ru', 'sr', 'sd', 'sk', 'sl', 'so',
        'es', 'sw', 'sv', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi',
        'cy', 'sq', 'ce', 'ny', 'si', 'sn', 'su', 'tg', 'tt', 'uz'
    }
    
    # Accepted tokens that mean: let the system auto-detect source language
    AUTO_DETECT_TOKENS: Set[str] = {"auto", "auto_detect", "auto-detect", "auto detect"}
    
    # Full language names mapping (for AI prompts and display)
    LANGUAGE_CODE_TO_NAME: Dict[str, str] = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
        'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
        'ar': 'Arabic', 'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'tr': 'Turkish',
        'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian',
        'af': 'Afrikaans', 'hy': 'Armenian', 'as': 'Assamese', 'az': 'Azerbaijani', 
        'be': 'Belarusian', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
        'hr': 'Croatian', 'cs': 'Czech', 'et': 'Estonian', 'fil': 'Filipino', 
        'fi': 'Finnish', 'gl': 'Galician', 'ka': 'Georgian', 'el': 'Greek', 
        'gu': 'Gujarati', 'ha': 'Hausa', 'he': 'Hebrew', 'hu': 'Hungarian',
        'is': 'Icelandic', 'id': 'Indonesian', 'ga': 'Irish', 'jv': 'Javanese',
        'kn': 'Kannada', 'kk': 'Kazakh', 'ky': 'Kyrgyz', 'lv': 'Latvian',
        'lt': 'Lithuanian', 'lb': 'Luxembourgish', 'mk': 'Macedonian', 'ms': 'Malay',
        'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali', 'ps': 'Pashto',
        'fa': 'Persian', 'pa': 'Punjabi', 'ro': 'Romanian', 'sr': 'Serbian',
        'sd': 'Sindhi', 'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali',
        'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai',
        'uk': 'Ukrainian', 'vi': 'Vietnamese', 'cy': 'Welsh', 'sq': 'Albanian',
        'ce': 'Chechen', 'ny': 'Chichewa', 'si': 'Sinhala', 'sn': 'Shona',
        'su': 'Sundanese', 'tg': 'Tajik', 'tt': 'Tatar', 'uz': 'Uzbek'
    }
    
    @classmethod
    def get_language_name(cls, code: str) -> str:
        """Get full language name from code (e.g., 'en' -> 'English')"""
        return cls.LANGUAGE_CODE_TO_NAME.get(code, code.title())
    
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
        if (language_lower in cls.DUBBING_SUPPORTED_CODES or 
            language_lower in cls.TRANSCRIPTION_SUPPORTED_CODES or
            language_lower in cls.ELEVENLABS_V3_LANGUAGES):
            return language_lower
        
        # Handle compound codes (en-US -> en) - but only if base code is supported
        if "-" in language_lower or "_" in language_lower:
            base_code = language_lower.split("-")[0].split("_")[0]
            if (base_code in cls.DUBBING_SUPPORTED_CODES or 
                base_code in cls.TRANSCRIPTION_SUPPORTED_CODES or
                base_code in cls.ELEVENLABS_V3_LANGUAGES):
                return base_code
        
        # Return as-is for 2-letter codes (might be valid for ElevenLabs)
        if len(language_lower) == 2:
            return language_lower
        
        # Return default if not found
        logger.warning(f"Unknown language input: {language}, defaulting to English")
        return "en"
    
    @classmethod
    def is_dubbing_supported(cls, language: str, model_type: str = "normal") -> bool:
        """
        Check if a language is supported for dubbing.
        - best (ElevenLabs): 70+ languages
        - medium (Fish): 17 languages
        - normal (Local): 17 languages
        """
        if not language:
            return False
        normalized = cls.normalize_language_input(language)
        
        if model_type == "best":
            return normalized in cls.ELEVENLABS_V3_LANGUAGES
        else:
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
    def get_supported_dubbing_languages(cls, model_type: str = "normal") -> Set[str]:
        """
        Get all supported language codes for dubbing based on model type.
        """
        if model_type == "best":
            return cls.ELEVENLABS_V3_LANGUAGES
        else:
            return cls.DUBBING_SUPPORTED_CODES
    
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
    
    @classmethod
    def get_assemblyai_language_code(cls, language: str):
        """
        Get AssemblyAI-compatible language code.
        AssemblyAI uses similar ISO codes but we map for compatibility.
        Returns None for auto_detect (AssemblyAI will auto-detect).
        """
        normalized = cls.normalize_language_input(language)
        
        if normalized == "auto_detect":
            return None
        
        assemblyai_mapping = {
            "en": "en",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "nl": "nl",
            "ru": "ru",
            "pl": "pl",
            "tr": "tr",
            "uk": "uk",
            "ar": "ar",
            "hi": "hi",
            "ja": "ja",
            "zh": "zh",
            "ko": "ko",
            "id": "id",
            "sv": "sv",
            "cs": "cs",
            "fi": "fi",
            "el": "el",
            "da": "da",
            "hu": "hu",
            "ro": "ro",
            "vi": "vi",
            "bg": "bg",
            "hr": "hr",
            "sk": "sk",
            "sl": "sl",
            "sr": "sr"
        }
        
        return assemblyai_mapping.get(normalized, "en")


# Global service instance
language_service = LanguageService()
