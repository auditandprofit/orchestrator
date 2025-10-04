#!/usr/bin/env bash
set -euo pipefail

if ! command -v doctl >/dev/null 2>&1; then
  echo "Error: doctl is not installed or not on PATH" >&2
  exit 1
fi

: "${DIGITALOCEAN_ACCESS_TOKEN?DIGITALOCEAN_ACCESS_TOKEN environment variable must be set}"
: "${DIGITALOCEAN_SSH_KEY_ID?DIGITALOCEAN_SSH_KEY_ID environment variable must be set}"

CONTEXT_NAME="test-context"
DROPLET_NAME="doctl-test-droplet"
REGION="nyc3"
SIZE="s-1vcpu-1gb"
IMAGE="ubuntu-22-04-x64"

# Configure the context so the test droplet uses scoped credentials.
if ! doctl auth list 2>/dev/null | grep -q "^${CONTEXT_NAME}\\b"; then
  doctl auth init --context "${CONTEXT_NAME}" --access-token "${DIGITALOCEAN_ACCESS_TOKEN}" >/dev/null
fi

doctl auth switch --context "${CONTEXT_NAME}" >/dev/null

# Launch a tiny test droplet.
doctl compute droplet create "${DROPLET_NAME}" \
  --size "${SIZE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --ssh-keys "${DIGITALOCEAN_SSH_KEY_ID}" \
  --tag-names "test" \
  --wait

# Output connection details for follow-up automation.
doctl compute droplet get "${DROPLET_NAME}" --format ID,Name,Status,PublicIPv4
