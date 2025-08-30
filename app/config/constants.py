"""
App Constants - Central location for all hardcoded values
All hardcoded values should be defined here for maintainability and configurability
"""

# Processing Constants
MAX_ATTEMPTS_DEFAULT = 90  # Default max attempts for background jobs (30 minutes with 20s intervals)
POLLING_INTERVAL_SECONDS = 20  # Default polling interval for background jobs
TIMEOUT_MINUTES = 30  # Default timeout for long-running jobs

MIN_CREDIT_CHECK_DURATION = 0.1  # Minimum duration to check credits for

# File Upload Constants
MAX_FILE_SIZE_MB = 1024  # 1GB max file size (user requirement)
MAX_SAFE_PROCESSING_SIZE_MB = 500  # 500MB safe processing limit (memory optimization)
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
CHUNK_SIZE_UPLOAD = 8192  # 8KB chunks for file uploads
LARGE_FILE_THRESHOLD_MB = 100  # Files above this use optimized processing

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
ERROR_FILE_TOO_LARGE = "File size exceeds safe processing limit (500MB). Please use a smaller file or contact support for assistance."

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
MAX_CONCURRENT_JOBS = 6  # Maximum concurrent processing jobs

# Audio Processing Constants
AUDIO_SAMPLE_RATE = 44100
AUDIO_DEFAULT_FORMAT = "wav"
VOICE_CLONE_TIMEOUT_SECONDS = 300  # 5 minutes timeout for voice cloning

# Queue Position Constants
MAX_QUEUE_POSITION_CHECKS = 5  # Maximum number of jobs to check queue position for (performance)
AVERAGE_JOB_PROCESSING_MINUTES = 3  # Average processing time per job for queue position estimation
PROCESSING_JOB_QUEUE_POSITION = 0  # Queue position for jobs currently being processed

# Database Query Constants
DEFAULT_QUERY_LIMIT = 50
MAX_QUERY_LIMIT = 1000