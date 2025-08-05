"""
Shared memory storage for upload status tracking across the application
"""

# Shared upload status memory storage
upload_status_memory = {}

def get_upload_status(job_id: str):
    """Get upload status for a job ID"""
    return upload_status_memory.get(job_id)

def set_upload_status(job_id: str, status_data: dict):
    """Set upload status for a job ID"""
    upload_status_memory[job_id] = status_data

def update_upload_status(job_id: str, updates: dict):
    """Update upload status for a job ID"""
    if job_id in upload_status_memory:
        upload_status_memory[job_id].update(updates)
    else:
        upload_status_memory[job_id] = updates

def delete_upload_status(job_id: str):
    """Delete upload status for a job ID"""
    if job_id in upload_status_memory:
        del upload_status_memory[job_id]

def job_exists(job_id: str) -> bool:
    """Check if job ID exists in upload status memory"""
    return job_id in upload_status_memory