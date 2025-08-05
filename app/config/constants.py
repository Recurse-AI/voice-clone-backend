"""
App Constants - Central location for all hardcoded values
All hardcoded values should be defined here for maintainability and configurability
"""

# Processing Constants
MAX_ATTEMPTS_DEFAULT = 180  # Default max attempts for background jobs (30 minutes with 10s intervals)
POLLING_INTERVAL_SECONDS = 10  # Default polling interval for background jobs
TIMEOUT_MINUTES = 30  # Default timeout for long-running jobs

# Credit Calculation Constants  
CREDITS_PER_MINUTE_SEPARATION = 1  # 1 credit per minute for audio separation
CREDITS_PER_MINUTE_DUB = 2  # 2 credits per minute for video dubbing
MIN_CREDIT_CHECK_DURATION = 0.1  # Minimum duration to check credits for

# File Upload Constants
MAX_FILE_SIZE_MB = 500  # 500MB max file size
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
CHUNK_SIZE_UPLOAD = 8192  # 8KB chunks for file uploads

# HTTP Response Constants
HTTP_STATUS_SUCCESS = 200
HTTP_STATUS_CREATED = 201
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_INTERNAL_ERROR = 500

# Error Messages
ERROR_USER_NOT_FOUND = "User not found"
ERROR_INVALID_TOKEN = "Invalid or expired token"
ERROR_INSUFFICIENT_CREDITS = "Insufficient credits"
ERROR_FILE_NOT_FOUND = "File not found"
ERROR_PROCESSING_FAILED = "Processing failed"
ERROR_AUTHENTICATION_REQUIRED = "Authentication required"

# Success Messages
MSG_PROCESSING_STARTED = "Processing started successfully"
MSG_CREDITS_DEDUCTED = "Credits deducted successfully"
MSG_FILE_UPLOADED = "File uploaded successfully"
MSG_JOB_COMPLETED = "Job completed successfully"

# Job Status Constants
JOB_STATUS_PENDING = "pending"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"

# Stripe/Payment Constants
MIN_PAYMENT_AMOUNT_USD = 1  # Minimum $1 payment
STRIPE_WEBHOOK_TOLERANCE = 300  # 5 minutes tolerance for webhook timestamps

# Background Thread Constants
THREAD_DAEMON_MODE = True  # All background threads should be daemon threads
MAX_THREAD_RETRY_ATTEMPTS = 3

# Cache/Memory Constants
SHARED_MEMORY_TTL_SECONDS = 3600  # 1 hour TTL for shared memory
MAX_CONCURRENT_JOBS = 10  # Maximum concurrent processing jobs

# Audio Processing Constants
AUDIO_SAMPLE_RATE = 44100
AUDIO_DEFAULT_FORMAT = "wav"
VOICE_CLONE_TIMEOUT_SECONDS = 300  # 5 minutes timeout for voice cloning

# Database Query Constants
DEFAULT_QUERY_LIMIT = 50
MAX_QUERY_LIMIT = 1000