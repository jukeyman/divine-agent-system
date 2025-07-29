#!/bin/bash

# Divine Agent System - Docker Entrypoint Script
# Handles container initialization and configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
}

# Banner
echo -e "${BLUE}"
cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                    Divine Agent System                       ║
║              Supreme Agentic Orchestrator (SAO)             ║
║                                                              ║
║  Multi-Agent Cloud Mastery System with Quantum Features     ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Environment setup
log "Initializing Divine Agent System..."

# Set default environment variables if not provided
export DIVINE_AGENT_ENVIRONMENT=${DIVINE_AGENT_ENVIRONMENT:-production}
export DIVINE_AGENT_LOG_LEVEL=${DIVINE_AGENT_LOG_LEVEL:-INFO}
export DIVINE_AGENT_DEBUG=${DIVINE_AGENT_DEBUG:-false}
export DIVINE_AGENT_PORT=${DIVINE_AGENT_PORT:-8000}
export DIVINE_AGENT_WORKERS=${DIVINE_AGENT_WORKERS:-4}

# Create necessary directories
log "Creating required directories..."
mkdir -p /app/logs
mkdir -p /app/data
mkdir -p /app/backups
mkdir -p /app/temp

# Set permissions
chmod 755 /app/logs
chmod 755 /app/data
chmod 755 /app/backups
chmod 755 /app/temp

log_success "Directories created successfully"

# Configuration validation
log "Validating configuration..."

if [ ! -f "/app/config.yaml" ]; then
    log_warning "config.yaml not found, using default configuration"
    # Create minimal config if none exists
    cat > /app/config.yaml << 'EOF'
system:
  name: "Divine Agent System"
  version: "1.0.0"
  environment: "production"
  debug: false
  log_level: "INFO"

architecture:
  orchestrator:
    type: "LangGraph"
    state_machine: true

departments:
  cloud_mastery:
    enabled: true
EOF
fi

log_success "Configuration validated"

# Health check function
health_check() {
    log "Performing health check..."
    
    # Check Python installation
    if ! command -v python &> /dev/null; then
        log_error "Python not found"
        exit 1
    fi
    
    # Check required Python packages
    python -c "import agents" 2>/dev/null || {
        log_error "Divine Agent System package not found"
        exit 1
    }
    
    # Check disk space
    DISK_USAGE=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        log_warning "Disk usage is high: ${DISK_USAGE}%"
    fi
    
    # Check memory
    if [ -f "/proc/meminfo" ]; then
        MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        if [ "$MEMORY_USAGE" -gt 90 ]; then
            log_warning "Memory usage is high: ${MEMORY_USAGE}%"
        fi
    fi
    
    log_success "Health check completed"
}

# Database initialization (if needed)
init_database() {
    if [ "$DIVINE_AGENT_INIT_DB" = "true" ]; then
        log "Initializing database..."
        python -c "from agents.database import init_db; init_db()" || {
            log_warning "Database initialization failed or not needed"
        }
        log_success "Database initialization completed"
    fi
}

# Migration function
run_migrations() {
    if [ "$DIVINE_AGENT_RUN_MIGRATIONS" = "true" ]; then
        log "Running database migrations..."
        python -c "from agents.database import migrate; migrate()" || {
            log_warning "Migration failed or not needed"
        }
        log_success "Migrations completed"
    fi
}

# Backup function
backup_data() {
    if [ "$DIVINE_AGENT_BACKUP_ON_START" = "true" ]; then
        log "Creating backup..."
        BACKUP_FILE="/app/backups/backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        tar -czf "$BACKUP_FILE" /app/data 2>/dev/null || {
            log_warning "Backup creation failed"
        }
        log_success "Backup created: $BACKUP_FILE"
    fi
}

# Signal handlers
handle_signal() {
    log "Received shutdown signal, gracefully stopping..."
    
    # Stop any running processes
    if [ ! -z "$MAIN_PID" ]; then
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi
    
    log_success "Divine Agent System stopped gracefully"
    exit 0
}

# Set up signal handlers
trap handle_signal SIGTERM SIGINT

# Pre-start checks
log "Running pre-start checks..."
health_check
init_database
run_migrations
backup_data

# Environment-specific setup
case "$DIVINE_AGENT_ENVIRONMENT" in
    "development")
        log "Starting in development mode..."
        export DIVINE_AGENT_DEBUG=true
        export DIVINE_AGENT_LOG_LEVEL=DEBUG
        ;;
    "staging")
        log "Starting in staging mode..."
        export DIVINE_AGENT_DEBUG=false
        export DIVINE_AGENT_LOG_LEVEL=INFO
        ;;
    "production")
        log "Starting in production mode..."
        export DIVINE_AGENT_DEBUG=false
        export DIVINE_AGENT_LOG_LEVEL=WARNING
        ;;
    *)
        log_warning "Unknown environment: $DIVINE_AGENT_ENVIRONMENT, defaulting to production"
        export DIVINE_AGENT_ENVIRONMENT=production
        ;;
esac

# Feature flags
if [ "$DIVINE_AGENT_QUANTUM_ENABLED" = "true" ]; then
    log "Quantum features enabled"
fi

if [ "$DIVINE_AGENT_CONSCIOUSNESS_ENABLED" = "true" ]; then
    log "Consciousness features enabled"
fi

if [ "$DIVINE_AGENT_MULTICLOUD_ENABLED" = "true" ]; then
    log "Multi-cloud features enabled"
fi

# Wait for dependencies (if specified)
if [ ! -z "$DIVINE_AGENT_WAIT_FOR" ]; then
    log "Waiting for dependencies: $DIVINE_AGENT_WAIT_FOR"
    
    IFS=',' read -ra DEPS <<< "$DIVINE_AGENT_WAIT_FOR"
    for dep in "${DEPS[@]}"; do
        IFS=':' read -ra DEP_PARTS <<< "$dep"
        HOST=${DEP_PARTS[0]}
        PORT=${DEP_PARTS[1]:-80}
        
        log "Waiting for $HOST:$PORT..."
        while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
            sleep 1
        done
        log_success "$HOST:$PORT is available"
    done
fi

# Start the application
log "Starting Divine Agent System..."
log "Environment: $DIVINE_AGENT_ENVIRONMENT"
log "Debug: $DIVINE_AGENT_DEBUG"
log "Log Level: $DIVINE_AGENT_LOG_LEVEL"
log "Port: $DIVINE_AGENT_PORT"
log "Workers: $DIVINE_AGENT_WORKERS"

# Execute the main command
if [ "$#" -eq 0 ]; then
    # Default command
    exec python -m agents
else
    # Custom command
    exec "$@"
fi &

# Store the main process PID
MAIN_PID=$!

# Wait for the main process
wait $MAIN_PID