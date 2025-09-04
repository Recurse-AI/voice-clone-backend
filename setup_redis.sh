#!/bin/bash

# Redis Setup Script for RunPod/Linux Environment
# Standalone script to install and configure Redis for RQ workers

echo "🔧 Redis Setup for Voice Cloning API"
echo "======================================"

# Function to check if Redis is running
check_redis() {
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is running"
        return 0
    else
        echo "❌ Redis is not running"
        return 1
    fi
}

# Check if Redis is already installed and running
if command -v redis-server &> /dev/null; then
    echo "📦 Redis is already installed"
    if check_redis; then
        echo "🎉 Redis is already running and ready!"
        exit 0
    else
        echo "🔄 Redis installed but not running, starting..."
    fi
else
    echo "📦 Installing Redis server..."
    
    # Update package list
    apt-get update -qq
    
    # Install Redis
    apt-get install -y redis-server
    
    if [ $? -eq 0 ]; then
        echo "✅ Redis installed successfully"
    else
        echo "❌ Failed to install Redis"
        exit 1
    fi
fi

# Configure Redis for production use
echo "⚙️ Configuring Redis..."

# Create Redis config directory if it doesn't exist
mkdir -p /etc/redis

# Basic Redis configuration
cat > /etc/redis/redis.conf << 'EOF'
# Basic Redis configuration for RQ workers
bind 127.0.0.1
port 6379
daemonize yes
pidfile /var/run/redis/redis-server.pid
logfile /var/log/redis/redis-server.log
dir /var/lib/redis

# Memory settings
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence settings (minimal for queue use)
save 900 1
save 300 10
save 60 10000

# Network settings
timeout 300
tcp-keepalive 300

# Security (basic)
# requirepass your_redis_password_here
EOF

# Create necessary directories
mkdir -p /var/run/redis
mkdir -p /var/log/redis
mkdir -p /var/lib/redis

# Set proper permissions
chown redis:redis /var/run/redis /var/log/redis /var/lib/redis 2>/dev/null || true

# Stop any existing Redis instances
echo "🛑 Stopping existing Redis instances..."
pkill -f redis-server 2>/dev/null || true
sleep 2

# Start Redis server
echo "🚀 Starting Redis server..."
redis-server /etc/redis/redis.conf

# Wait for Redis to start
sleep 3

# Verify Redis is running
if check_redis; then
    echo ""
    echo "🎉 Redis setup completed successfully!"
    echo ""
    echo "📊 Redis Status:"
    echo "  - Server: Running on 127.0.0.1:6379"
    echo "  - Memory: 2GB max"
    echo "  - Config: /etc/redis/redis.conf"
    echo "  - Logs: /var/log/redis/redis-server.log"
    echo ""
    echo "💡 Useful commands:"
    echo "  - Check status: redis-cli ping"
    echo "  - Monitor: redis-cli monitor"
    echo "  - Info: redis-cli info"
    echo "  - Stop: redis-cli shutdown"
    echo ""
else
    echo "❌ Redis setup failed!"
    echo "Check logs: tail -f /var/log/redis/redis-server.log"
    exit 1
fi

# Set environment variable for RQ
export REDIS_URL="redis://127.0.0.1:6379"
echo "✅ REDIS_URL set to: $REDIS_URL"

echo "🔥 Redis is ready for RQ workers!"
