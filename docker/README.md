# Docker Configuration Files

This directory contains configuration files for running HiveFrame in Docker containers.

## Files

### prometheus.yml
Prometheus configuration for monitoring HiveFrame multi-colony deployments.

**Features:**
- Monitors all colony instances
- Tracks dashboard metrics
- Collects system-level metrics
- 15-second scrape interval for real-time monitoring

**Endpoints monitored:**
- `colony-us-east:9090` - US East colony metrics
- `colony-us-west:9090` - US West colony metrics
- `colony-eu-central:9090` - EU Central colony metrics
- `dashboard:9090` - Dashboard metrics

**Usage:**
This configuration is automatically used by the Prometheus container in the main `docker-compose.yml` file.

## Extending Monitoring

To add custom metrics or additional colonies:

1. Edit `prometheus.yml`
2. Add new targets to the appropriate job
3. Restart the Prometheus container:
   ```bash
   docker compose restart prometheus
   ```

## Custom Dashboards

To integrate with Grafana or other visualization tools:

1. Point your visualization tool to `http://localhost:9090`
2. Use the Prometheus data source
3. Create dashboards using PromQL queries

Example queries:
```promql
# Worker utilization by colony
hiveframe_worker_utilization

# Task processing rate
rate(hiveframe_tasks_completed[5m])

# Error rate
rate(hiveframe_tasks_failed[5m]) / rate(hiveframe_tasks_total[5m])
```

## See Also

- [Main Docker Documentation](../DOCKER.md)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [HiveFrame Monitoring](../src/hiveframe/monitoring/)
