# OilCast signed model-channel administration

This controls only OilCast public model selection. It never changes a Google Play track,
an Android package identity, personal invoice data, or any cryptographic model weights.

## Force a channel from GitHub Actions

Open Actions -> **OilCast admin - force cloud prediction model** -> Run workflow
(on the main branch). Choose:
- `auto`: remove backend override; each user keeps their own app channel preference
- `stable`: force the latest signed Adaptive v6 public model
- `neural-v7`: force the latest separately published signed Neural v7 test model

The job signs a purpose-specific policy using the existing model-signing secret
and publishes only the public RSA-signed envelope to
`https://raw.githubusercontent.com/lins58308/oilcast-models/main/model-control-envelope.json`.
Only the signed policy changes effective model selection. Untrusted repository status
pointers cannot enforce a channel. The policy expires after 14 days and requires
renewal. An unavailable or invalid test model fails safely to the signed stable model.
The user's model preference remains stored locally even during an override.

## One-time safety configuration

Create the GitHub Environment `oilcast-model-admin` in this training repository,
restrict deployment to main, and configure required reviewer approval for
administrative changes. Existing repository Actions Secrets
`MODEL_PUBLISH_TOKEN` and `MODEL_SIGNING_KEY_B64` must already be configured
for the ordinary model publisher. Do NOT copy secret values into inputs, commits,
logs, issues, or chat. The Android Play upload signing key is unrelated and must
never be installed here.

The signed policy uses purpose `model-channel-control-v1`, expiry, repository
binding, and the same RSA public key pinned in the app. The App uses a signed,
fresh, eligible model only; it does not download executable code.

## Validation

Check the Actions run, then verify that public
`model-control-envelope.json` was updated and that the app's Settings page shows
the signed backend override and the actual active channel. This is not verified
until the admin workflow has run and a device has downloaded the policy.
