"""
Process Logger for Video Processing API

Clean logging system that tracks each step of video processing.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class ProcessLogger:
    """Clean logger for tracking video processing steps"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.current_audio_id = None
        self.current_log_file = None
        self.start_time = None
        self.steps = []
    
    def start_processing(self, audio_id: str, video_url: str, parameters: Dict[str, Any]) -> None:
        """Start logging for a new processing session"""
        self.current_audio_id = audio_id
        self.start_time = time.time()
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"process_{audio_id}_{timestamp}.log"
        self.current_log_file = self.log_dir / log_filename
        
        # Initialize log
        self.steps = []
        self._write_header(video_url, parameters)
    
    def log_step(self, step_name: str, status: str, details: Optional[Dict[str, Any]] = None, 
                 duration: Optional[float] = None) -> None:
        """Log a processing step"""
        if not self.current_audio_id:
            return
            
        step_time = time.time()
        elapsed_total = step_time - self.start_time if self.start_time else 0
        
        step_data = {
            "step": step_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "elapsed_total": round(elapsed_total, 2),
            "step_duration": round(duration, 2) if duration else None,
            "details": details or {}
        }
        
        self.steps.append(step_data)
        self._write_step(step_data)
    
    def get_current_log_path(self) -> Optional[str]:
        """Get the current log file path"""
        return str(self.current_log_file) if self.current_log_file else None
    
    def finish_processing(self, success: bool, total_duration: Optional[float] = None, 
                         final_details: Optional[Dict[str, Any]] = None) -> None:
        """Finish logging and write summary"""
        if not self.current_audio_id:
            return
            
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "success": success,
            "total_duration": round(total_duration or total_time, 2),
            "total_steps": len(self.steps),
            "completed_steps": len([s for s in self.steps if s["status"] == "completed"]),
            "failed_steps": len([s for s in self.steps if s["status"] == "failed"]),
            "final_details": final_details or {}
        }
        
        self._write_summary(summary)
        
        # Reset for next processing
        self.current_audio_id = None
        self.current_log_file = None
        self.start_time = None
        self.steps = []
    
    def _write_header(self, video_url: str, parameters: Dict[str, Any]) -> None:
        """Write log header"""
        header = f"""
=== VIDEO PROCESSING LOG ===
Audio ID: {self.current_audio_id}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video URL: {video_url}
Parameters: {json.dumps(parameters, indent=2)}

=== PROCESSING STEPS ===
"""
        
        with open(self.current_log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def _write_step(self, step_data: Dict[str, Any]) -> None:
        """Write a single step to log"""
        status_symbol = "✓" if step_data["status"] == "completed" else "✗" if step_data["status"] == "failed" else "⟳"
        
        step_line = f"\n[{step_data['elapsed_total']}s] {status_symbol} {step_data['step'].upper()} - {step_data['status']}"
        
        if step_data["step_duration"]:
            step_line += f" (took {step_data['step_duration']}s)"
            
        if step_data["details"]:
            step_line += f"\n    Details: {json.dumps(step_data['details'], indent=4)}"
        
        step_line += "\n" + "-" * 50
        
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(step_line)
    
    def _write_summary(self, summary: Dict[str, Any]) -> None:
        """Write processing summary"""
        summary_text = f"""

=== PROCESSING SUMMARY ===
Status: {'SUCCESS' if summary['success'] else 'FAILED'}
Total Duration: {summary['total_duration']}s
Total Steps: {summary['total_steps']}
Completed: {summary['completed_steps']}
Failed: {summary['failed_steps']}
End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Final Details:
{json.dumps(summary['final_details'], indent=2)}

=== END LOG ===
"""
        
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(summary_text)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        if not self.current_audio_id:
            return {"status": "inactive"}
            
        return {
            "status": "active",
            "audio_id": self.current_audio_id,
            "elapsed_time": round(time.time() - self.start_time, 2) if self.start_time else 0,
            "total_steps": len(self.steps),
            "completed_steps": len([s for s in self.steps if s["status"] == "completed"]),
            "current_step": self.steps[-1]["step"] if self.steps else None
        }


# Global logger instance
process_logger = ProcessLogger() 