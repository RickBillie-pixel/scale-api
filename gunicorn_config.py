# gunicorn_config.py
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
backlog = 2048

# Worker processes - optimized for CPU-bound scale calculations
workers = min(multiprocessing.cpu_count() + 1, 4)  # Cap at 4 workers
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeout settings
timeout = 300  # 5 minutes for complex calculations
graceful_timeout = 60
keepalive = 5

# Restart workers periodically to prevent memory bloat
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'scale-api'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# Worker temp directory
worker_tmp_dir = "/dev/shm"

# Preload app for better performance
preload_app = True

# Stats
statsd_host = None
statsd_prefix = "scale_api"
