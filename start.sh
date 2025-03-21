#!/bin/bash

echo "Starting ML Workflow Tool..."

# Start backend server
echo "Starting backend server..."
cd src/backend && python app.py &
cd ../..

# Start frontend server
echo "Starting frontend server..."
npm start &

echo "ML Workflow Tool started. Please wait for the application to open in your browser."
echo "If the application doesn't open automatically, navigate to http://localhost:3000" 