#!/bin/bash

# Aiberm Provider Test Script
# Usage: ./test_aiberm.sh <API_KEY>

API_KEY="${1:-your-api-key}"
BASE_URL="${2:-http://localhost:6600}"

echo "================================"
echo "Aiberm Provider Test Script"
echo "Base URL: $BASE_URL"
echo "================================"
echo ""

# Test 1: List models
echo "[Test 1] List models"
echo "--------------------"
curl -s "$BASE_URL/v1/models" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL/v1/models"
echo ""
echo ""

# Test 2: Basic chat (aiberm provider)
echo "[Test 2] Basic chat with aiberm/gpt-4o"
echo "--------------------------------------"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "aiberm/gpt-4o",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_completion_tokens": 100
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""
echo ""

# Test 3: Streaming chat
echo "[Test 3] Streaming chat with aiberm/gpt-4o"
echo "------------------------------------------"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "aiberm/gpt-4o",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "max_completion_tokens": 100,
    "stream": true
  }'
echo ""
echo ""

# Test 4: Test with max_tokens (should also work)
echo "[Test 4] Test with max_tokens parameter"
echo "---------------------------------------"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "aiberm/gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""
echo ""

# Test 5: Claude provider (for comparison)
echo "[Test 5] Claude provider (for comparison)"
echo "-----------------------------------------"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "claude/claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 50
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""

echo "================================"
echo "Tests completed"
echo "================================"
