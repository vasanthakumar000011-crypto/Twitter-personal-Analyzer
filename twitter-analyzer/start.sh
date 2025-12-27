#!/bin/bash

# Twitter Persona Analyzer - Quick Start Script
echo "🐦 Twitter Persona Analyzer & Generator"
echo "========================================"

# Check if .env exists, if not create from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp env.example .env
    echo "⚠️  Please edit .env with your OpenAI API key or local AI settings"
    echo "   You can continue without it to test with mock data"
    echo ""
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "🚀 Starting Twitter Persona Analyzer..."
echo "This may take a few minutes on first run (downloading images)..."
echo ""

# Build and start the application
docker-compose up --build -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "✅ SUCCESS! Twitter Persona Analyzer is running!"
    echo ""
    echo "🌐 Open your browser to: http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo ""
    echo "🛑 To stop: docker-compose down"
    echo "📊 To view logs: docker-compose logs -f"
else
    echo ""
    echo "⚠️  Application might still be starting..."
    echo "🌐 Try opening: http://localhost:8000"
    echo "📊 Check logs with: docker-compose logs -f"
fi 