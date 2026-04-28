# HF Spaces Docker template handles pip install + streamlit automatically.
# Only install system deps needed for RDKit.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libxft2 \
    libfreetype6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*
