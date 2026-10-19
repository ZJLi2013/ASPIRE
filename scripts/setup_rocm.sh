#!/usr/bin/env bash
# LIBERO-Pro (SAM3 + PyRoki) on an AMD Instinct GPU with ROCm 6.4.
#
#   scripts/setup_rocm.sh install    build .venv-libero, .venv-pyroki and a ROCm .venv-sam3
#   scripts/setup_rocm.sh smoke      start the providers, run doctor and object_swap_seed1 in both init modes
#   scripts/setup_rocm.sh services   start the providers in the foreground
#
# Tested inside rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4. SAM3 weights come from the
# gated Hugging Face repository unless SAM3_CHECKPOINT points at a local sam3.pt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/rocm"
ROCM_INDEX=https://download.pytorch.org/whl/rocm6.4
PROVIDERS=sam3,pyroki

export PATH="$HOME/.local/bin:$PATH"
# The image preconfigures a regional package mirror; resolve from PyPI instead.
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL UV_INDEX_URL UV_EXTRA_INDEX_URL
export UV_DEFAULT_INDEX=https://pypi.org/simple

install() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  scripts/bootstrap_libero.sh
  scripts/bootstrap_provider.sh pyroki

  # bootstrap_provider.sh sam3 installs the locked cu128 wheels; take torch from the ROCm index instead.
  git submodule update --init third_party/sam3
  uv venv --allow-existing --python python3 .venv-sam3
  printf '%s\n' 'numpy==1.26.4' 'transformers<5.0' "triton ; sys_platform == 'never'" > "$OUT/sam3-overrides.txt"
  VIRTUAL_ENV="$ROOT/.venv-sam3" uv pip install \
    --extra-index-url "$ROCM_INDEX" --index-strategy unsafe-best-match \
    --override "$OUT/sam3-overrides.txt" \
    "torch==2.9.1+rocm6.4" "torchvision==0.24.1+rocm6.4" \
    -e third_party/sam3 -e . \
    einops "decord>=0.6" "psutil>=5.9" "open_clip_torch>=2.24" "opencv-python-headless<4.13" \
    "pillow>=10" "pycocotools>=2.0" "fastapi>=0.115" "uvicorn>=0.34" "pydantic>=2.7" "setuptools<81"
  .venv-sam3/bin/python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpus', torch.cuda.device_count())"
}

services() {
  exec scripts/supervise_services.sh --providers "$PROVIDERS" --log-dir "$OUT/services" --ready-timeout 600
}

smoke() {
  source .venv-libero/bin/activate
  export LIBERO_CONFIG_PATH="$ROOT/.libero" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  # The image's Mesa cannot create an EGL device display on AMD; render offscreen with OSMesa.
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  unset EGL_PLATFORM

  scripts/supervise_services.sh --providers "$PROVIDERS" --log-dir "$OUT/services" \
    --ready-timeout 600 > "$OUT/supervisor.txt" 2>&1 &
  trap "kill $! 2>/dev/null" EXIT
  until [[ "$(status 8114)$(status 8116)" == 200200 ]]; do sleep 5; done

  cap-harness doctor --providers "$PROVIDERS" --output "$OUT/smoke/doctor.json" || true
  for mode in saved seeded; do
    cap-harness run --benchmark libero-pro --suite libero_object_swap --task-id 0 --seed 1 \
      --init-mode "$mode" --program examples/libero/object_swap_seed1.py --output-root "$OUT/smoke/$mode"
  done
  python - "$OUT/smoke" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
doctor = json.loads((root / "doctor.json").read_text())
print("doctor fail:", [c["name"] for c in doctor["checks"] if c["status"] == "fail"])
for outcome in sorted(root.glob("*/**/outcome.json")):
    o = json.loads(outcome.read_text())
    print(outcome.relative_to(root).parts[0], o["termination_reason"], o["steps_executed"], "steps")
EOF
}

status() { curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$1/openapi.json"; }

cd "$ROOT"
mkdir -p "$OUT/smoke"
case "${1:-}" in
  install | smoke | services) "$1" ;;
  *) sed -n '2,6p' "$0" >&2; exit 2 ;;
esac
