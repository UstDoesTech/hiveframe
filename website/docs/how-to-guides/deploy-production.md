---
sidebar_position: 10
---

# Deploy to Production

This guide shows you how to deploy HiveFrame applications to production environments.

## Prerequisites

- HiveFrame application working in development
- Production environment (VM, container, or cloud)
- Monitoring infrastructure (optional but recommended)

## Installation

### Using pip

Install HiveFrame with production dependencies:

```bash
pip install hiveframe[production]
```

This includes:
- Kafka support
- PostgreSQL connectors
- HTTP clients
- Prometheus metrics

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run application
CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t my-hiveframe-app .
docker run -d --name hiveframe my-hiveframe-app
```

## Configuration

### Environment Variables

Set production configuration via environment variables:

```bash
export HIVE_NUM_WORKERS=16
export HIVE_ABANDONMENT_LIMIT=30
export HIVE_DANCE_THRESHOLD=0.4
export HIVE_LOG_LEVEL=INFO
export HIVE_METRICS_PORT=9090
```

### Configuration File

Create `config.yaml`:

```yaml
hive:
  num_workers: 16
  employed_ratio: 0.5
  onlooker_ratio: 0.4
  scout_ratio: 0.1
  abandonment_limit: 30
  max_cycles: 200
  dance_threshold: 0.4

logging:
  level: INFO
  format: json
  
monitoring:
  metrics_port: 9090
  enable_profiling: true
  
resilience:
  retry_max_attempts: 3
  circuit_breaker_threshold: 10
```

Load in your application:

```python
import yaml
from hiveframe import HiveFrame

with open('config.yaml') as f:
    config = yaml.safe_load(f)

hive = HiveFrame(**config['hive'])
```

## Deployment Strategies

### Single Server

Deploy on a single powerful server:

```python
# main.py
from hiveframe import HiveDataFrame, col
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Load data
    df = HiveDataFrame.from_csv('data.csv')
    
    # Process
    result = (df
        .filter(col('amount') > 100)
        .groupBy('category')
        .agg(sum_agg(col('amount')))
    )
    
    # Save results
    result.to_csv('output.csv')
    logging.info("Processing complete")

if __name__ == '__main__':
    main()
```

Run with systemd:

```ini
# /etc/systemd/system/hiveframe.service
[Unit]
Description=HiveFrame Application
After=network.target

[Service]
Type=simple
User=hiveframe
WorkingDirectory=/opt/hiveframe
ExecStart=/usr/bin/python3 /opt/hiveframe/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable hiveframe
sudo systemctl start hiveframe
sudo systemctl status hiveframe
```

### Kubernetes

Deploy on Kubernetes for scalability:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hiveframe-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hiveframe
  template:
    metadata:
      labels:
        app: hiveframe
    spec:
      containers:
      - name: hiveframe
        image: my-hiveframe-app:latest
        env:
        - name: HIVE_NUM_WORKERS
          value: "16"
        - name: HIVE_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "2"
          limits:
            memory: "4Gi"
            cpu: "4"
        ports:
        - containerPort: 9090
          name: metrics
---
apiVersion: v1
kind: Service
metadata:
  name: hiveframe-metrics
spec:
  selector:
    app: hiveframe
  ports:
  - port: 9090
    targetPort: 9090
    name: metrics
```

Apply:

```bash
kubectl apply -f deployment.yaml
```

### Docker Compose

For multi-container setups:

```yaml
# docker-compose.yml
version: '3.8'

services:
  hiveframe:
    build: .
    environment:
      - HIVE_NUM_WORKERS=16
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - kafka
      - postgres
    ports:
      - "9090:9090"
    volumes:
      - ./data:/data
    restart: unless-stopped

  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_PASSWORD=secret
    volumes:
      - postgres-data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

volumes:
  postgres-data:
  prometheus-data:
```

Start:

```bash
docker-compose up -d
```

## Monitoring

### Prometheus Metrics

Export metrics to Prometheus:

```python
from hiveframe.monitoring import get_registry
from prometheus_client import start_http_server

# Start metrics server
start_http_server(9090)

# Your application code
hive = HiveFrame(num_workers=16)
# ... processing ...
```

Configure Prometheus (`prometheus.yml`):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hiveframe'
    static_configs:
      - targets: ['localhost:9090']
```

### Logging

Configure structured logging:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
```

### Health Checks

Implement health endpoints:

```python
from flask import Flask, jsonify
from hiveframe import HiveFrame

app = Flask(__name__)
hive = HiveFrame(num_workers=16)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'workers': hive.get_worker_count(),
        'temperature': hive.get_colony_temperature()
    })

@app.route('/metrics')
def metrics():
    return jsonify(hive.get_metrics())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Security

### Secure Configuration

- Store secrets in environment variables or secret management systems
- Use TLS for network communication
- Restrict file permissions
- Run with minimal privileges

Example with secrets:

```python
import os
from hiveframe import HiveDataFrame

# Get credentials from environment
db_password = os.environ.get('DB_PASSWORD')
kafka_token = os.environ.get('KAFKA_TOKEN')

# Never hardcode secrets!
```

### Network Security

Configure firewall rules:

```bash
# Allow only necessary ports
sudo ufw allow 9090/tcp  # Metrics
sudo ufw allow 8080/tcp  # Health check
sudo ufw enable
```

## Performance Tuning

### Worker Configuration

```python
hive = HiveFrame(
    num_workers=16,           # Match CPU cores
    employed_ratio=0.5,       # 50% exploit
    onlooker_ratio=0.4,       # 40% reinforce
    scout_ratio=0.1,          # 10% explore
    abandonment_limit=30,     # Higher for production
    max_cycles=200            # More cycles for complex tasks
)
```

### Resource Limits

Set memory and CPU limits:

```python
# In Docker
docker run -m 4g --cpus=4 my-hiveframe-app

# In Kubernetes (see deployment.yaml above)
```

## Troubleshooting

### Check Logs

```bash
# Docker
docker logs hiveframe

# Kubernetes
kubectl logs deployment/hiveframe-app

# Systemd
journalctl -u hiveframe -f
```

### Monitor Metrics

Visit Prometheus: `http://localhost:9091`

Query useful metrics:
- `hiveframe_workers_active` - Active workers
- `hiveframe_partitions_processed` - Completed partitions
- `hiveframe_colony_temperature` - System load

### Common Issues

**High memory usage:**
- Reduce `num_workers`
- Process data in smaller batches
- Check for memory leaks in processing functions

**Slow processing:**
- Increase `num_workers`
- Tune `abandonment_limit`
- Check data partitioning

**Worker failures:**
- Check logs for exceptions
- Verify resource availability
- Review abandonment metrics

## Backup and Recovery

### Data Backup

```bash
# Backup results
tar -czf backup-$(date +%Y%m%d).tar.gz /opt/hiveframe/data/

# Restore
tar -xzf backup-20240115.tar.gz -C /opt/hiveframe/
```

### State Recovery

HiveFrame is stateless by design. Recovery is automatic through the abandonment mechanism.

## Next Steps

- **Monitor Applications** - Set up comprehensive monitoring
- **Scale Your Application** - Horizontal scaling strategies
- **Handle Errors** - Error handling patterns
- **Secure Deployment** - Advanced security

## Related Topics

- **Colony Architecture** - Understand the system design
- **Monitoring API** - Metrics and logging reference
- **Configuration Reference** - All configuration options
