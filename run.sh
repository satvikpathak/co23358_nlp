#!/usr/bin/env bash
# Simple run script for Tone Transformer
# Starts both backend and frontend with a single command

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_DIR="$PROJECT_DIR/.venv"
PID_FILE_BACKEND="$PROJECT_DIR/backend.pid"
PID_FILE_FRONTEND="$PROJECT_DIR/frontend.pid"

# Ports
BACKEND_PORT=8001
FRONTEND_PORT=8080

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Tone Transformer - NLP Project${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    echo -e "${YELLOW}Killing process on port $port...${NC}"
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    sleep 1
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    
    # Kill backend
    if [ -f "$PID_FILE_BACKEND" ]; then
        backend_pid=$(cat "$PID_FILE_BACKEND")
        if kill -0 "$backend_pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping backend (PID: $backend_pid)...${NC}"
            kill "$backend_pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$PID_FILE_BACKEND"
    fi
    
    # Kill frontend
    if [ -f "$PID_FILE_FRONTEND" ]; then
        frontend_pid=$(cat "$PID_FILE_FRONTEND")
        if kill -0 "$frontend_pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping frontend (PID: $frontend_pid)...${NC}"
            kill "$frontend_pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$PID_FILE_FRONTEND"
    fi
    
    # Ensure ports are free
    check_port $BACKEND_PORT && kill_port $BACKEND_PORT
    check_port $FRONTEND_PORT && kill_port $FRONTEND_PORT
    
    echo -e "${GREEN}Services stopped.${NC}"
}

# Function to setup environment
setup_environment() {
    echo -e "${BLUE}Setting up environment...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Install/upgrade dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r "$PROJECT_DIR/requirements.txt" > /dev/null 2>&1
    
    echo -e "${GREEN}Environment ready!${NC}"
}

# Function to start backend
start_backend() {
    echo -e "${BLUE}Starting backend on port $BACKEND_PORT...${NC}"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Start backend
    cd "$PROJECT_DIR"
    nohup uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT > backend.log 2>&1 &
    backend_pid=$!
    echo $backend_pid > "$PID_FILE_BACKEND"
    
    # Wait for backend to start
    echo -e "${YELLOW}Waiting for backend to start...${NC}"
    for i in {1..30}; do
        if check_port $BACKEND_PORT; then
            echo -e "${GREEN}✓ Backend started successfully! (PID: $backend_pid)${NC}"
            echo -e "${GREEN}  URL: http://localhost:$BACKEND_PORT${NC}"
            echo -e "${GREEN}  Logs: $PROJECT_DIR/backend.log${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}✗ Backend failed to start. Check backend.log for details.${NC}"
    return 1
}

# Function to start frontend
start_frontend() {
    echo -e "${BLUE}Starting frontend on port $FRONTEND_PORT...${NC}"
    
    # Check if Python http.server is available
    cd "$FRONTEND_DIR"
    nohup python3 -m http.server $FRONTEND_PORT > "$PROJECT_DIR/frontend.log" 2>&1 &
    frontend_pid=$!
    echo $frontend_pid > "$PID_FILE_FRONTEND"
    
    # Wait for frontend to start
    echo -e "${YELLOW}Waiting for frontend to start...${NC}"
    for i in {1..10}; do
        if check_port $FRONTEND_PORT; then
            echo -e "${GREEN}✓ Frontend started successfully! (PID: $frontend_pid)${NC}"
            echo -e "${GREEN}  URL: http://localhost:$FRONTEND_PORT${NC}"
            echo -e "${GREEN}  Logs: $PROJECT_DIR/frontend.log${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}✗ Frontend failed to start. Check frontend.log for details.${NC}"
    return 1
}

# Function to show status
show_status() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Service Status${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ -f "$PID_FILE_BACKEND" ] && kill -0 $(cat "$PID_FILE_BACKEND") 2>/dev/null; then
        echo -e "${GREEN}✓ Backend:  Running on http://localhost:$BACKEND_PORT${NC}"
        echo -e "  PID: $(cat $PID_FILE_BACKEND)"
    else
        echo -e "${RED}✗ Backend:  Not running${NC}"
    fi
    
    if [ -f "$PID_FILE_FRONTEND" ] && kill -0 $(cat "$PID_FILE_FRONTEND") 2>/dev/null; then
        echo -e "${GREEN}✓ Frontend: Running on http://localhost:$FRONTEND_PORT${NC}"
        echo -e "  PID: $(cat $PID_FILE_FRONTEND)"
    else
        echo -e "${RED}✗ Frontend: Not running${NC}"
    fi
    
    echo ""
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start    Start both backend and frontend (default)"
    echo "  stop     Stop both backend and frontend"
    echo "  restart  Restart both services"
    echo "  status   Show service status"
    echo "  logs     Show logs (use Ctrl+C to exit)"
    echo "  help     Show this help message"
    echo ""
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}Showing logs (Ctrl+C to exit)...${NC}"
    echo ""
    tail -f "$PROJECT_DIR/backend.log" "$PROJECT_DIR/frontend.log" 2>/dev/null || echo "No logs found"
}

# Main logic
COMMAND="${1:-start}"

case "$COMMAND" in
    start)
        # Stop any existing services
        stop_services
        
        # Setup environment
        setup_environment
        
        # Start services
        start_backend || exit 1
        sleep 2
        start_frontend || exit 1
        
        # Show status
        show_status
        
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  All services started successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "${BLUE}Frontend:${NC} http://localhost:$FRONTEND_PORT"
        echo -e "${BLUE}Backend API:${NC} http://localhost:$BACKEND_PORT"
        echo -e "${BLUE}API Docs:${NC} http://localhost:$BACKEND_PORT/docs"
        echo ""
        echo -e "${YELLOW}To stop services, run:${NC} ./run.sh stop"
        echo -e "${YELLOW}To view logs, run:${NC} ./run.sh logs"
        echo ""
        ;;
    
    stop)
        stop_services
        ;;
    
    restart)
        echo -e "${BLUE}Restarting services...${NC}"
        stop_services
        sleep 2
        $0 start
        ;;
    
    status)
        show_status
        ;;
    
    logs)
        show_logs
        ;;
    
    help|--help|-h)
        show_usage
        ;;
    
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
