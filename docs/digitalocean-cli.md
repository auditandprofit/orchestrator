# DigitalOcean CLI Setup

This repository includes a helper script for configuring the [DigitalOcean CLI (`doctl`)](https://docs.digitalocean.com/reference/doctl/) and launching a tiny test droplet.

## Installing `doctl`

The CLI is not available in the default Ubuntu package repositories, so download the latest release archive directly from GitHub:

```bash
LATEST_TAG=$(curl -s https://api.github.com/repos/digitalocean/doctl/releases/latest | jq -r '.tag_name')
curl -sL "https://github.com/digitalocean/doctl/releases/download/${LATEST_TAG}/doctl-${LATEST_TAG#v}-linux-amd64.tar.gz" -o doctl.tar.gz
tar -xzf doctl.tar.gz
sudo mv doctl /usr/local/bin/
doctl version
```

## Authenticating and launching a test droplet

1. Generate a DigitalOcean personal access token with read/write scopes and export it:

   ```bash
   export DIGITALOCEAN_ACCESS_TOKEN="<your-token>"
   ```

2. Determine the SSH key ID that should be added to the droplet and export it:

   ```bash
   export DIGITALOCEAN_SSH_KEY_ID="$(doctl compute ssh-key list --format ID --no-header | head -n 1)"
   ```

   Replace the command above if you need a specific key.

3. Run the helper script to configure a `doctl` context and create a tiny test droplet:

   ```bash
   ./scripts/doctl-test-droplet.sh
   ```

   The script creates a `s-1vcpu-1gb` droplet in `nyc3`, waits for it to become active, and prints its ID and IPv4 address. The droplet is tagged with `test` so it can be cleaned up with:

   ```bash
   doctl compute droplet delete --tag-name test --force
   ```

> **Note:** The script requires valid credentials and SSH keys. Without them DigitalOcean will reject the API call, so it cannot be executed successfully within the automated test environment.
