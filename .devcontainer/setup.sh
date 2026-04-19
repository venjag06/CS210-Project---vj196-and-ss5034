#!/usr/bin/env bash
# One-time setup for the Codespace.
# Runs on first boot via devcontainer.json's postCreateCommand.
set -euo pipefail

echo "=========================================="
echo "  CS210 Rental Fairness :: Codespace Init "
echo "=========================================="

# --- 1. Install PostgreSQL ---------------------------------------------------
echo "[1/4] Installing PostgreSQL..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    postgresql postgresql-contrib

# --- 2. Start the Postgres service ------------------------------------------
echo "[2/4] Starting PostgreSQL service..."
sudo service postgresql start
# Give it a moment to accept connections
sleep 2

# --- 3. Create the project DB user and database -----------------------------
echo "[3/4] Creating DB user 'vscode' and database 'rental_fairness'..."

# Create user if it doesn't already exist
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vscode'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE USER vscode WITH SUPERUSER PASSWORD 'vscode';"

# Create database if it doesn't already exist
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='rental_fairness'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE DATABASE rental_fairness OWNER vscode;"

# --- 4. Install Python dependencies and apply schema ------------------------
echo "[4/4] Installing Python packages + applying schema..."
pip install --upgrade pip
pip install -r requirements.txt

if [ -f db/schema.sql ]; then
    PGPASSWORD=vscode psql -h localhost -U vscode -d rental_fairness -f db/schema.sql
    echo "    Schema applied."
else
    echo "    (db/schema.sql not found yet -- skipping.)"
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Connect with:"
echo "    psql -h localhost -U vscode -d rental_fairness"
echo "    password: vscode"
echo "=========================================="