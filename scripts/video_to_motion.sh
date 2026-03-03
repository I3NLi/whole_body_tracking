#!/usr/bin/env bash
set -euo pipefail

# Canonical one-click entrypoint required by ops.
# Delegates to existing bundle pipeline.
exec /home/hiyio/whole_body_tracking/scripts/video_to_motion_bundle.sh "$@"
