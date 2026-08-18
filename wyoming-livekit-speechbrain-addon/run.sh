#!/usr/bin/env sh
# shellcheck shell=ash
set -e

CONFIG_PATH=/data/options.json
WAKEWORDS_DIR=/share/wakewords

if [ ! -f "$CONFIG_PATH" ]; then
    echo "FATAL: $CONFIG_PATH not found - is this running as a HA add-on?" >&2
    exit 1
fi

ENABLE_SPK=$(jq -r '.enable_speaker_verification // false' "$CONFIG_PATH")
SPK_THRESHOLD=$(jq -r '.speaker_threshold // 0.1' "$CONFIG_PATH")
PREDICT_INTERVAL=$(jq -r '.predict_interval_seconds // 0.5' "$CONFIG_PATH")
MID_SENT=$(jq -r '.enable_mid_sentence_detection // true' "$CONFIG_PATH")
MAX_SAMPLES=$(jq -r '.max_samples // 1000' "$CONFIG_PATH")
SENS_VALUE=$(jq -r '.sensitivity // 0.37' "$CONFIG_PATH")
SAVE_SAMPLE_THRESHOLD=$(jq -r '.save_sample_threshold // empty' "$CONFIG_PATH")
DEBUG=$(jq -r '.debug_logging // false' "$CONFIG_PATH")

for OLD_KEY in enable_amplitude_normalization livekit_amplitude_normalization wake_word_models wake_words speaker_profiles speaker_model; do
    if jq -e ".$OLD_KEY" "$CONFIG_PATH" >/dev/null 2>&1; then
        echo "WARN: config key '$OLD_KEY' is no longer used (auto-detected now). Update via Configuration tab to clear this warning." >&2
    fi
done

mkdir -p /data/speechbrain /data/hf-cache "$WAKEWORDS_DIR"

if ! ls "$WAKEWORDS_DIR"/*.onnx >/dev/null 2>&1 && [ -f /app/models/sputnik.onnx ]; then
    echo "INFO: no wake-word models in $WAKEWORDS_DIR - seeding bundled sputnik.onnx" >&2
    cp /app/models/sputnik.onnx "$WAKEWORDS_DIR/sputnik.onnx"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ONNXRUNTIME_INTRA_OP_NUM_THREADS=1
export ONNXRUNTIME_INTER_OP_NUM_THREADS=1

ARGS=""
MODEL_COUNT=0
for M in "$WAKEWORDS_DIR"/*.onnx; do
    [ -f "$M" ] || continue
    ARGS="$ARGS --model $M --sensitivity $SENS_VALUE"
    MODEL_COUNT=$((MODEL_COUNT + 1))
done

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "FATAL: no *.onnx files found in $WAKEWORDS_DIR" >&2
    echo "  Drop ONNX wake-word classifier files into $WAKEWORDS_DIR/ (one per wake word)." >&2
    echo "  Wake-word names are auto-derived from filename stems (e.g. sputnik.onnx -> 'sputnik')." >&2
    exit 2
fi

if [ "$ENABLE_SPK" = "true" ]; then
    ARGS="$ARGS --speaker-threshold $SPK_THRESHOLD"
    ARGS="$ARGS --speaker-savedir /data/speechbrain"
fi

if [ "$MID_SENT" = "false" ]; then
    ARGS="$ARGS --no-mid-sentence-detection"
fi

ARGS="$ARGS --max-samples $MAX_SAMPLES"
ARGS="$ARGS --predict-interval-seconds $PREDICT_INTERVAL"

if [ -n "$SAVE_SAMPLE_THRESHOLD" ]; then
    ARGS="$ARGS --save-sample-threshold $SAVE_SAMPLE_THRESHOLD"
fi

MAX_SAMPLES_PER_SOURCE=$(jq -r '.max_samples_per_source[]? // empty' "$CONFIG_PATH" 2>/dev/null || true)
for SPEC in $MAX_SAMPLES_PER_SOURCE; do
    [ -z "$SPEC" ] && continue
    ARGS="$ARGS --max-samples-source $SPEC"
done

if [ "$DEBUG" = "true" ]; then
    ARGS="$ARGS --debug"
fi

echo "Starting Wyoming LiveKit+SpeechBrain on 0.0.0.0:10400"
echo "Auto-discovered $MODEL_COUNT wake-word model(s) from $WAKEWORDS_DIR"
echo "Args: $ARGS"

# shellcheck disable=SC2086
exec python3 -m wyoming_livekit_speechbrain \
    --host 0.0.0.0 \
    --port 10400 \
    $ARGS
