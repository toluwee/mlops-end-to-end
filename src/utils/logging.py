import logging
import json
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields if they exist
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'model_version'):
            log_entry['model_version'] = record.model_version

        return json.dumps(log_entry)

def setup_logging(level: str = "INFO"):
    """Setup structured logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Apply JSON formatter to all handlers
    for handler in logging.root.handlers:
        handler.setFormatter(JSONFormatter())
