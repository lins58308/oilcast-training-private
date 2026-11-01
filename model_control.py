#!/usr/bin/env python3
"""Publish a short-lived, RSA-signed channel override; never expose signing secrets.

Only the trusted GitHub Actions workflow invokes this. Users can select stable
or neural-v7 in the app. A valid signed forced channel takes precedence.
"""
import base64
import datetime
import json
import os
import pathlib
import re
import subprocess
import tempfile
import urllib.error
import urllib.request

PUBLIC_REPO = "lins58308/oilcast-models"
API_URL = f"https://api.github.com/repos/{PUBLIC_REPO}/contents/model-control-envelope.json"
ALLOWED = {"auto", "stable", "neural-v7"}


def main():
    channel = os.environ.get("MODEL_FORCE_CHANNEL", "")
    token = os.environ.get("MODEL_PUBLISH_TOKEN", "")
    key_b64 = os.environ.get("MODEL_SIGNING_KEY_B64", "")
    if channel not in ALLOWED:
        raise ValueError("Unexpected model channel; allowed: auto/stable/neural-v7")
    if not token or not key_b64:
        raise RuntimeError("Signing or publication secrets are not configured")
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise RuntimeError("Only an administrator-initiated workflow can change the override")
    now = datetime.datetime.now(datetime.timezone.utc)
    issued = now.isoformat().replace("+00:00", "Z")
    expires = (now + datetime.timedelta(days=14)).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema": 1,
        "purpose": "model-channel-control-v1",
        "repository": PUBLIC_REPO,
        "channel": channel,
        "issuedAt": issued,
        "expiresAt": expires
    }
    payload = json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="oilcast-control-") as tmp:
        pem_file = pathlib.Path(tmp, "private.pem")
        payload_file = pathlib.Path(tmp, "manifest.json")
        signature_file = pathlib.Path(tmp, "signature.bin")
        pem_file.write_bytes(base64.b64decode(key_b64, validate=True))
        pem_file.chmod(0o600)
        payload_file.write_bytes(payload)
        subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(pem_file),
                        "-out", str(signature_file), str(payload_file)],
                       check=True, capture_output=True)
        subprocess.run(["openssl", "dgst", "-sha256", "-verify", "model-public.txt",
                        "-signature", str(signature_file), str(payload_file)],
                       check=True, capture_output=True)
        signature = signature_file.read_bytes()
    envelope = {
        "schema": 1,
        "keyId": "oilcast-model-v1",
        "signatureAlgorithm": "SHA256withRSA",
        "manifestBase64": base64.b64encode(payload).decode("ascii"),
        "signatureBase64": base64.b64encode(signature).decode("ascii")
    }
    contents = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
        "Authorization": f"Bearer {token}",
        "User-Agent": "OilCastModelControl/1"
    }
    existing_sha = None
    try:
        req = urllib.request.Request(API_URL, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=30) as response:
            existing_sha = json.load(response)["sha"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    body = {
        "message": "Update signed OilCast model-channel control",
        "branch": "main",
        "content": base64.b64encode(contents).decode("ascii")
    }
    if existing_sha:
        body["sha"] = existing_sha
    req = urllib.request.Request(
        API_URL, headers={**headers, "Content-Type": "application/json"},
        method="PUT", data=json.dumps(body).encode("utf-8"))
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status not in (200, 201):
            raise RuntimeError("Unexpected status when updating signed control")
    # Never print the key material, token, raw envelope, or signature.
    print("Signed channel control published:", channel, "; expires:", expires)


if __name__ == "__main__":
    main()
