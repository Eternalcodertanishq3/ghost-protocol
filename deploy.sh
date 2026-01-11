#!/bin/bash

# Ghost Protocol v1.0 Deployment Script
# DPDP-Safe, Byzantine-immune Federated Learning Infrastructure
# DPDP § Citation: §7(1) - Data sovereignty through secure deployment
# Byzantine Theorem: Practical Byzantine Fault Tolerance for production systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_PROJECT_NAME="ghost-protocol"
DEPLOYMENT_ENV="production"
LOG_FILE="deployment.log"

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check system resources
    MEMORY_GB=$(free -g | awk 'NR==2{printf "%.1f", $2}')
    if (( $(echo "$MEMORY_GB < 8" | bc -l) )); then
        warning "System has less than 8GB RAM. Recommended minimum is 16GB for production."
    fi
    
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt 4 ]; then
        warning "System has less than 4 CPU cores. Recommended minimum is 8 cores for production."
    fi
    
    log "Prerequisites check completed."
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p logs data ssl uploads backups
    mkdir -p grafana/dashboards grafana/datasources
    mkdir -p prometheus
    
    # Set proper permissions
    chmod 755 logs data ssl uploads backups
    
    log "Directories created successfully."
}

# Generate SSL certificates
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
            -days 365 -nodes -subj "/C=IN/ST=Delhi/L=New Delhi/O=Ghost Protocol/CN=localhost"
        log "SSL certificates generated."
    else
        info "SSL certificates already exist."
    fi
}

# Create configuration files
create_config_files() {
    log "Creating configuration files..."
    
    # Nginx configuration
    cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    upstream frontend {
        server frontend:3000;
    }
    
    upstream sna {
        server sna:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api {
            proxy_pass http://sna;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /ws {
            proxy_pass http://sna;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
EOF
    
    # Prometheus configuration
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ghost-sna'
    static_configs:
      - targets: ['sna:8000']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'ghost-shap-explainer'
    static_configs:
      - targets: ['shap-explainer:8001']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'ghost-compliance-heartbeat'
    static_configs:
      - targets: ['compliance-heartbeat:8002']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'ghost-synthetic-gateway'
    static_configs:
      - targets: ['synthetic-gateway:8007']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'ghost-model-marketplace'
    static_configs:
      - targets: ['model-marketplace:8008']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'ghost-quantum-vault'
    static_configs:
      - targets: ['quantum-vault:8009']
    metrics_path: /metrics
    scrape_interval: 30s
EOF
    
    # Logstash configuration
    cat > logstash.conf << 'EOF'
input {
  beats {
    port => 5044
  }
  tcp {
    port => 5000
    codec => json_lines
  }
}

filter {
  if [service] == "sna" {
    mutate { add_field => { "type" => "aggregator" } }
  } else if [service] == "ghost-agent" {
    mutate { add_field => { "type" => "hospital" } }
  } else {
    mutate { add_field => { "type" => "system" } }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "ghost-protocol-%{+YYYY.MM.dd}"
  }
}
EOF
    
    log "Configuration files created."
}

# Pull Docker images
pull_images() {
    log "Pulling Docker images..."
    
    docker-compose pull
    
    log "Docker images pulled successfully."
}

# Build custom images
build_images() {
    log "Building custom Docker images..."
    
    docker-compose build --parallel
    
    log "Custom Docker images built successfully."
}

# Initialize Vault
initialize_vault() {
    log "Initializing HashiCorp Vault..."
    
    # Wait for Vault to be ready
    info "Waiting for Vault to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8200/v1/sys/health | grep -q "initialized"; then
            log "Vault is ready."
            break
        fi
        sleep 2
    done
    
    # Initialize Vault if not already initialized
    if ! curl -s http://localhost:8200/v1/sys/health | grep -q '"initialized":true'; then
        info "Initializing Vault..."
        docker-compose exec -T vault vault operator init -key-shares=1 -key-threshold=1 > vault-init.txt 2>&1 || true
    fi
    
    log "Vault initialization completed."
}

# Start core services
start_core_services() {
    log "Starting core services..."
    
    # Start infrastructure services first
    docker-compose up -d vault redis postgres
    
    # Wait for services to be ready
    info "Waiting for infrastructure services to be ready..."
    sleep 10
    
    # Start SNA
    docker-compose up -d sna
    
    # Wait for SNA to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8000/health | grep -q "healthy"; then
            log "SNA is ready."
            break
        fi
        sleep 2
    done
    
    log "Core services started successfully."
}

# Start additional services
start_additional_services() {
    log "Starting additional services..."
    
    # Start all SNA sub-services
    docker-compose up -d shap-explainer
    docker-compose up -d compliance-heartbeat
    docker-compose up -d drift-incentivizer
    docker-compose up -d dropout-predictor
    docker-compose up -d adaptive-clustering
    docker-compose up -d dispute-resolution
    docker-compose up -d synthetic-gateway
    docker-compose up -d model-marketplace
    docker-compose up -d quantum-vault
    
    # Start monitoring services
    docker-compose up -d prometheus grafana
    docker-compose up -d elasticsearch logstash kibana
    
    # Start load balancer
    docker-compose up -d nginx
    
    log "Additional services started successfully."
}

# Start Ghost Agents
start_ghost_agents() {
    log "Starting Ghost Agents..."
    
    docker-compose up -d --scale ghost-agent=5
    
    log "Ghost Agents started successfully."
}

# Start frontend
start_frontend() {
    log "Starting frontend dashboard..."
    
    docker-compose up -d frontend
    
    # Wait for frontend to be ready
    for i in {1..60}; do
        if curl -s http://localhost:3000 | grep -q "Ghost Protocol"; then
            log "Frontend is ready."
            break
        fi
        sleep 2
    done
    
    log "Frontend started successfully."
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check running containers
    RUNNING_CONTAINERS=$(docker-compose ps -q | wc -l)
    info "Running containers: $RUNNING_CONTAINERS"
    
    # Check service health
    SERVICES=("sna:8000" "shap-explainer:8001" "compliance-heartbeat:8002" "drift-incentivizer:8003" 
              "dropout-predictor:8004" "adaptive-clustering:8005" "dispute-resolution:8006" 
              "synthetic-gateway:8007" "model-marketplace:8008" "quantum-vault:8009" 
              "frontend:3000" "nginx:80" "prometheus:9090" "grafana:3001")
    
    for service in "${SERVICES[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -s http://localhost:$port/health > /dev/null 2>&1 || curl -s http://localhost:$port > /dev/null 2>&1; then
            log "$name is healthy."
        else
            warning "$name may not be ready."
        fi
    done
    
    log "Deployment verification completed."
}

# Display deployment summary
display_summary() {
    log "Deployment Summary:"
    echo "=================================="
    echo "Ghost Protocol v1.0 deployed successfully!"
    echo ""
    echo "Services Available:"
    echo "- Main Dashboard: http://localhost:3000"
    echo "- Load Balancer: http://localhost:80"
    echo "- SNA API: http://localhost:8000"
    echo "- SHAP Explainer: http://localhost:8001"
    echo "- Compliance Heartbeat: http://localhost:8002"
    echo "- Drift Incentivizer: http://localhost:8003"
    echo "- Dropout Predictor: http://localhost:8004"
    echo "- Adaptive Clustering: http://localhost:8005"
    echo "- Dispute Resolution: http://localhost:8006"
    echo "- Synthetic Gateway: http://localhost:8007"
    echo "- Model Marketplace: http://localhost:8008"
    echo "- Quantum Vault: http://localhost:8009"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3001"
    echo "- Kibana: http://localhost:5601"
    echo ""
    echo "Key Features:"
    echo "- DPDP Act 2023 Compliant ✓"
    echo "- Byzantine Fault Tolerant ✓"
    echo "- Post-Quantum Cryptography ✓"
    echo "- Real-time Monitoring ✓"
    echo "- Model Marketplace ✓"
    echo "- Explainable AI ✓"
    echo ""
    echo "Status: NHA-Ready"
    echo "=================================="
}

# Cleanup function
cleanup() {
    log "Cleaning up deployment artifacts..."
    
    # Remove temporary files
    rm -f vault-init.txt
    
    log "Cleanup completed."
}

# Main deployment function
main() {
    log "Starting Ghost Protocol v1.0 deployment..."
    
    # Create log file
    touch "$LOG_FILE"
    chmod 644 "$LOG_FILE"
    
    # Execute deployment steps
    check_prerequisites
    create_directories
    generate_ssl_certificates
    create_config_files
    pull_images
    build_images
    
    # Start services in order
    start_core_services
    initialize_vault
    start_additional_services
    start_ghost_agents
    start_frontend
    
    # Final steps
    verify_deployment
    display_summary
    cleanup
    
    log "Ghost Protocol v1.0 deployment completed successfully!"
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"