# 🐝 HiveFrame Docker Deployment Guide

This guide explains how to deploy HiveFrame in Docker containers, enabling easy testing of multi-colony systems and production deployments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Multi-Colony Setup](#multi-colony-setup)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+
- At least 4GB RAM available for Docker
- Ports 8080 (dashboard) and 9090 (metrics) available

## Quick Start

### 1. Single Colony (Development)

The simplest way to get started with HiveFrame in Docker:

```bash
# Build the image
docker-compose -f docker-compose.simple.yml build

# Start a single colony with interactive Python shell
docker-compose -f docker-compose.simple.yml up

# In another terminal, connect to the running container
docker exec -it hiveframe-single python3
```

Then in the Python shell:

```python
from hiveframe import create_hive, HiveDataFrame

# Create a hive with 8 workers
hive = create_hive(num_workers=8)

# Process some data
data = list(range(100))
results = hive.map(data, lambda x: x * 2)
print(results)

# Use DataFrame API
df = HiveDataFrame.from_list([
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
])
df.show()
```

### 2. Multi-Colony Federation (Testing)

To test multi-colony coordination:

```bash
# Build and start all services
docker-compose up -d

# View logs from all colonies
docker-compose logs -f

# View logs from a specific colony
docker-compose logs -f colony-us-east

# Access the dashboard
open http://localhost:8080
```

This starts:
- 3 colonies (us-east, us-west, eu-central) with 8 workers each
- 1 dashboard for monitoring (port 8080)
- 1 Prometheus instance for metrics (port 9090)

## Deployment Options

### Option 1: Pre-built Image

Build the HiveFrame image:

```bash
docker build -t hiveframe:latest .
```

Build with a specific version:

```bash
docker build --build-arg HIVEFRAME_VERSION=1.0.0 -t hiveframe:1.0.0 .
```

Run a single container:

```bash
docker run -it --rm \
  -p 8080:8080 \
  -p 9090:9090 \
  -e HIVE_NUM_WORKERS=8 \
  hiveframe:latest python3
```

### Option 2: Docker Compose (Recommended)

#### Simple Single Colony

```bash
docker-compose -f docker-compose.simple.yml up
```

#### Multi-Colony Federation

```bash
docker-compose up
```

#### With Custom Configuration

Create a `.env` file:

```env
HIVE_NUM_WORKERS=16
HIVE_EMPLOYED_RATIO=0.5
HIVE_ONLOOKER_RATIO=0.4
HIVE_SCOUT_RATIO=0.1
HIVE_ABANDONMENT_LIMIT=10
HIVE_MAX_CYCLES=100
```

Then start:

```bash
docker-compose --env-file .env up
```

## Configuration

### Environment Variables

Configure HiveFrame behavior with these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HIVE_COLONY_NAME` | Name of the colony | `default` |
| `HIVE_COLONY_REGION` | Geographic region | `local` |
| `HIVE_NUM_WORKERS` | Number of bee workers | `8` |
| `HIVE_EMPLOYED_RATIO` | Ratio of employed bees | `0.5` |
| `HIVE_ONLOOKER_RATIO` | Ratio of onlooker bees | `0.4` |
| `HIVE_SCOUT_RATIO` | Ratio of scout bees | `0.1` |
| `HIVE_ABANDONMENT_LIMIT` | Task abandonment threshold | `10` |
| `HIVE_MAX_CYCLES` | Maximum processing cycles | `100` |

### Bee Worker Roles

The three bee roles must sum to 1.0:

- **Employed Bees (0.5)**: Exploit known good tasks
- **Onlooker Bees (0.4)**: Reinforce high-quality tasks
- **Scout Bees (0.1)**: Explore new tasks

## Multi-Colony Setup

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Docker Network                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │ Colony       │  │ Colony       │  │ Colony     ││
│  │ US-East      │  │ US-West      │  │ EU-Central ││
│  │ 8 workers    │  │ 8 workers    │  │ 8 workers  ││
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────┘│
│         │                 │                  │       │
│         └─────────────────┴──────────────────┘       │
│                           │                          │
│                  ┌────────▼─────────┐                │
│                  │   Dashboard      │                │
│                  │   Port 8080      │                │
│                  └──────────────────┘                │
└─────────────────────────────────────────────────────┘
```

### Running Federated Workloads

Execute code across multiple colonies:

```bash
# Start the federation
docker-compose up -d

# Run a federated example
docker-compose exec colony-us-east python3 /app/examples/demo_phase2_federation.py

# Or run custom code
docker-compose exec colony-us-east python3 << 'EOF'
from hiveframe.distributed import HiveFederation, FederatedHive

# Create federation
federation = HiveFederation(name="docker-federation")

# Register colonies
for region in ["us-east", "us-west", "eu-central"]:
    hive = FederatedHive(
        name=f"{region}-hive",
        endpoint=f"colony-{region}:9000",
        workers=8
    )
    federation.register_hive(hive)

print(f"Federation ready with {len(federation.hives)} colonies")
EOF
```

### Scaling Colonies

Scale individual colonies:

```bash
# Scale US-East colony to 3 instances
docker-compose up -d --scale colony-us-east=3

# Scale all colonies
docker-compose up -d --scale colony-us-east=3 --scale colony-us-west=3
```

## Monitoring

### Dashboard

Access the web dashboard at http://localhost:8080 to view:

- Colony health and status
- Worker activity (employed/onlooker/scout)
- Waggle dance activity
- Task quality metrics
- Real-time performance graphs

### Prometheus Metrics

Access Prometheus at http://localhost:9090 for:

- Colony-level metrics
- Worker utilization
- Task processing rates
- System resource usage

Example queries:

```promql
# Average worker utilization
avg(hiveframe_worker_utilization)

# Task processing rate
rate(hiveframe_tasks_completed[5m])

# Failed task ratio
rate(hiveframe_tasks_failed[5m]) / rate(hiveframe_tasks_total[5m])
```

### Container Logs

View logs from all services:

```bash
# All services
docker-compose logs -f

# Specific colony
docker-compose logs -f colony-us-east

# Last 100 lines
docker-compose logs --tail=100 colony-us-west
```

## Troubleshooting

### Container Won't Start

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs colony-us-east

# Rebuild without cache
docker-compose build --no-cache
```

### Port Already in Use

```bash
# Check what's using port 8080
lsof -i :8080  # macOS/Linux
netstat -ano | findstr :8080  # Windows

# Use different ports
docker-compose up -d -p 8081:8080
```

### Out of Memory

```bash
# Check Docker memory usage
docker stats

# Increase Docker memory limit (Docker Desktop settings)
# Or reduce number of workers
docker-compose up -d -e HIVE_NUM_WORKERS=4
```

### Colonies Can't Communicate

```bash
# Check network
docker network inspect hiveframe_hive-network

# Test connectivity
docker-compose exec colony-us-east ping colony-us-west

# Restart network
docker-compose down
docker-compose up -d
```

### Dashboard Not Accessible

```bash
# Check if dashboard is running
docker-compose ps dashboard

# Check dashboard logs
docker-compose logs dashboard

# Test from inside container
docker-compose exec dashboard curl http://localhost:8080
```

## Advanced Usage

### Custom Dockerfile

For production with specific dependencies:

```dockerfile
FROM hiveframe:latest

# Install additional dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn

# Copy custom code
COPY ./my_app /app/my_app

# Set custom entry point
CMD ["python3", "/app/my_app/main.py"]
```

### Kubernetes Deployment

HiveFrame includes built-in Kubernetes support. See the Kubernetes operator in `src/hiveframe/k8s/` for production deployments.

### Persistent Storage

Mount volumes for data persistence:

```yaml
volumes:
  - ./data:/app/data
  - ./logs:/app/logs
  - ./config:/app/config
```

### Production Hardening

1. **Use specific image tags**: Replace `latest` with version tags
2. **Resource limits**: Set memory/CPU limits in docker-compose
3. **Health checks**: Configure appropriate health check intervals
4. **Logging**: Use a centralized logging solution
5. **Secrets**: Use Docker secrets or environment files
6. **Network security**: Use firewall rules and encrypted connections

## Next Steps

- [Read the main README](README.md) for HiveFrame concepts
- [Explore examples](examples/) for usage patterns
- [View Kubernetes deployment](src/hiveframe/k8s/) for production
- [Check monitoring setup](docker/prometheus.yml) for observability

## Support

For issues and questions:
- GitHub Issues: https://github.com/hiveframe/hiveframe/issues
- Documentation: https://hiveframe.readthedocs.io
