# Two stages on purpose. Image size is most of a Cloud Run cold start, and the compilers
# needed to install scientific wheels are not needed to run them, so they stay in the builder
# and never reach the deployed image.

# ---------- builder ----------
FROM python:3.11-slim AS builder

# git is required because some requirements install straight from GitHub; build-essential
# covers any dependency that has no prebuilt wheel for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Copied alone so that editing application code does not invalidate the dependency layer,
# which is the slowest part of every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime ----------
FROM python:3.11-slim

# Curl is kept for container health checks and nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

# Streamlit writes a config and cache at runtime, and the container filesystem is read-only
# in places, so both are pointed at a writable location. This is the same class of failure
# that broke the deployed research platform with a PermissionError on a read-only venv path.
# HOME drives where Streamlit looks for ~/.streamlit and where libraries place caches, so
# pointing it at a writable path is what keeps both off any read-only part of the image.
ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PORT=8080

EXPOSE 8080

# Cloud Run injects PORT and the container must bind it on 0.0.0.0. Binding localhost is the
# single most common reason a working app serves nothing once deployed.
CMD streamlit run app.py \
      --server.port=${PORT} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false
