#!/bin/bash

# Production deployment script
set -e

echo "Starting deployment..."

# Pull latest image
docker pull $DOCKERHUB_USERNAME/credit-default-model:latest

# Stop current containers
docker-compose down

# Start new containers
docker-compose up -d

# Health check
sleep 30
curl -f http://localhost:8000/health || exit 1

echo "Deployment completed successfully!"