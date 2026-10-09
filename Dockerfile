# syntax=docker/dockerfile:1.4

FROM --platform=linux/amd64 condaforge/miniforge3:24.3.0-0

ARG VERSION=2.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    SSL_QALAS_VERSION=${VERSION} \
    PYTHONNOUSERSITE=1 \
    HOME=/tmp \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# ----------------------------------------------------------------------
# System dependencies
# ----------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
       curl \
       build-essential \
   && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# Application source
# ----------------------------------------------------------------------

WORKDIR /opt/ssl_qalas

COPY . /opt/ssl_qalas

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------

RUN conda env create -f environment.yml \
   && conda run -n ssl_qalas_crossvendor \
       pip install --no-cache-dir -e .

# ----------------------------------------------------------------------
# surfa
# ----------------------------------------------------------------------

RUN conda run -n ssl_qalas_crossvendor \
       pip install --no-cache-dir "numpy<2" surfa

# ----------------------------------------------------------------------
# Build patchelf OUTSIDE the Conda environment
# ----------------------------------------------------------------------

RUN curl -fsSL \
       https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0.tar.gz \
       -o /tmp/patchelf.tar.gz \
   && mkdir -p /tmp/patchelf \
   && tar -xzf /tmp/patchelf.tar.gz \
       --strip-components=1 \
       -C /tmp/patchelf \
   && cd /tmp/patchelf \
   && ./configure \
   && make -j"$(nproc)" \
   && make install \
   && /usr/local/bin/patchelf --version \
   && rm -rf /tmp/patchelf /tmp/patchelf.tar.gz

# ----------------------------------------------------------------------
# Fix Torch executable-stack issue
# ----------------------------------------------------------------------

RUN find \
       /opt/conda/envs/ssl_qalas_crossvendor/lib/python3.9/site-packages/torch/lib \
       -name '*.so*' \
       -type f \
       -exec /usr/local/bin/patchelf --clear-execstack {} \; \
   && conda run -n ssl_qalas_crossvendor python -c \
       "import surfa, numpy, torch; print('surfa', surfa.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)" \
   && conda clean -afy

# ----------------------------------------------------------------------
# Minimal FreeSurfer stand-in
# ----------------------------------------------------------------------

ENV FREESURFER_HOME=/opt/fslite

RUN mkdir -p /opt/fslite/bin /opt/fslite/models \
   && curl -fsSL \
       -o /opt/fslite/mri_synthstrip.py \
       https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip \
   && curl -fsSL \
       -o /opt/fslite/models/synthstrip.1.pt \
       https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt \
   && touch /opt/fslite/SetUpFreeSurfer.sh

# ----------------------------------------------------------------------
# mri_synthstrip wrapper
# ----------------------------------------------------------------------

RUN cat > /opt/fslite/bin/mri_synthstrip <<'EOF'
#!/bin/bash

exec python /opt/fslite/mri_synthstrip.py \
   --model /opt/fslite/models/synthstrip.1.pt \
   "$@"
EOF

# ----------------------------------------------------------------------
# Minimal mri_info
# ----------------------------------------------------------------------

RUN cat > /opt/fslite/bin/mri_info <<'EOF'
#!/usr/bin/env python

import sys
import nibabel as nib

if len(sys.argv) != 3 or sys.argv[1] != "--nframes":
    sys.exit(
        "mri_info shim only supports: "
        "mri_info --nframes FILE"
    )

shape = nib.load(sys.argv[2]).shape

print(shape[3] if len(shape) > 3 else 1)
EOF

# ----------------------------------------------------------------------
# Minimal mri_convert
# ----------------------------------------------------------------------

RUN cat > /opt/fslite/bin/mri_convert <<'EOF'
#!/usr/bin/env python

import sys
import nibabel as nib

args = sys.argv[1:]

if "--frame" not in args:
    sys.exit(
        "mri_convert shim only supports: "
        "IN --frame N OUT"
    )

i = args.index("--frame")

frame = int(args[i + 1])

del args[i:i + 2]

if len(args) != 2:
    sys.exit(
        "mri_convert shim only supports: "
        "IN --frame N OUT"
    )

src, dst = args

img = nib.load(src)

data = img.dataobj[..., frame]

nib.save(
    nib.Nifti1Image(
        data,
        img.affine,
        img.header,
    ),
    dst,
)
EOF

# ----------------------------------------------------------------------
# Permissions
# ----------------------------------------------------------------------

RUN chmod +x \
       /opt/fslite/bin/* \
       /opt/ssl_qalas/run.py

# ----------------------------------------------------------------------
# Runtime environment
# ----------------------------------------------------------------------

ENV PATH=/opt/conda/envs/ssl_qalas_crossvendor/bin:/opt/conda/bin:/opt/fslite/bin:$PATH

ENTRYPOINT ["python", "/opt/ssl_qalas/run.py"]