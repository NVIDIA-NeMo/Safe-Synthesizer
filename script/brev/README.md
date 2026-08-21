<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

### NeMo Safe Synthesizer Brev Launchable

This directory holds the provisioning script and recorded console settings for the
[NVIDIA Brev](https://brev.nvidia.com) Launchable, a one-click GPU environment for trying
NeMo Safe Synthesizer without setting up CUDA, drivers, or Python locally.

Nothing in this directory is executed by the repo or by CI. A Brev Launchable is
configured in the Brev web console, and the setup script is pasted into a form field
there. This directory exists so that configuration is versioned and reviewable rather
than living only in a browser. When you change `setup.sh` or `welcome.md`, you must also
update it in the console for the change to take effect.

### Files

- `setup.sh`: Pasted into the Launchable's Setup Script field. Installs the CUDA
  build of Safe Synthesizer into a dedicated venv, provisions Triton's runtime compiler,
  registers the venv as the default Jupyter kernel, and drops the tutorial notebooks in
  `$HOME`.
- `welcome.md`: Added to the Launchable's Source files so it renders on the Launchable
  webpage and appears as the customer's `$HOME/welcome.md`.

### Console configuration

Recreate the Launchable at <https://brev.nvidia.com> → Launchables → Create
Launchable with these settings.

| Section | Field | Value |
| ------- | ----- | ----- |
| Software | Runtime mode | VM Mode |
| Software | Install Jupyter on the host | Enabled |
| Software | Run a Setup Script | Enabled, contents of `setup.sh` |
| Software | Image ID | Leave blank |
| Source | Code source | `welcome.md` |
| Hardware | GPU | 1× 80 GiB VRAM, single GPU |
| Hardware | Disk | 200 GiB or more -- not resizable after creation |
| Network | Ports | 8888, named `jupyter` |
| Access | Visibility | Anyone with the link |

Multi-GPU instances are pointless here: training is single-GPU only, so a second GPU
adds cost and no capability.

### Launch parameters

All optional. Per Brev's guidance, do not put credential values in parameter defaults.

| Name | Type | Purpose |
| ---- | ---- | ------- |
| `NSS_INFERENCE_KEY` | Text | Unused on this branch (reserved for PII replacement v3). |
| `HF_TOKEN` | Text | Only needed for gated Hugging Face models. |

Brev passes these to `setup.sh` as environment variables.

Deliberately kept to two. A dataset-URL parameter and a model-prewarm toggle were both
tried and removed: a customer can fetch a URL with one `!curl` line in a notebook, and
prewarming moves the model download earlier without making it shorter, while hiding the
progress bar the notebook would otherwise show. Every parameter is visible to every
deployer, so the bar for adding one is high.

### What the customer gets

`$HOME` is the JupyterLab file browser root, so only what a customer needs is visible
there. Everything operational is a dotfile, which the browser hides by default.

```text
$HOME/
  tutorials/                    the three tutorial notebooks and their datasets
  welcome.md                    where to start, rendered on double-click
  SETUP-IN-PROGRESS.md          present only while setup is running or after failure

  .nss-venv/                    cu129 venv, registered as the default kernel
  .cache/huggingface/           model cache (Hugging Face's default location)
  .nss-env.sh                   PATH and VIRTUAL_ENV, sourced from .bashrc
  .nss-setup.log                full provisioning log
```

The script deliberately does not create a data folder. Customers put their files
wherever they like and pass the path to `with_data_source()`. Bring-your-own-data
options, in rough order of convenience: drag and drop into the JupyterLab file browser;
`!curl -O <url>` in a notebook cell; or `brev shell` plus `scp` for large files.

### Implementation notes

These are the non-obvious constraints the script works around. They were each found the
hard way on a real instance.

- Package indexes are derived at runtime, not hardcoded. Index URLs are
  install-time configuration rather than wheel metadata, so the installer must supply
  them -- and they have to match the release being installed, not this repo's `main`.
  The script resolves the latest version from the PyPI JSON API, fetches that tag's
  `pyproject.toml`, and reads the CUDA index URLs out of it, then pins the install to
  that exact version so the two cannot drift. Selection uses the CUDA extra in each
  index's name or URL and includes indexes referenced by `[tool.uv.sources]` for that
  extra. The source lookup matters for variant-neutral indexes such as
  `https://flashinfer.ai/whl/`, while the URL lookup handles names that changed between
  releases. The parse runs inside a process substitution and therefore cannot fail the
  script, so the count of discovered indexes is what validates it.
- uv is installed from a checksum-verified tarball, not `curl | sh`. The
  `astral.sh/install.sh` path logs `no checksums to verify`, so nothing validated what
  it downloaded. The script fetches the pinned release tarball, compares it against the
  published `.sha256`, and installs the binaries -- matching the GPG-verified mise
  install in `tools/install-mise.sh`.
- A placeholder file marks the home directory while setup runs. JupyterLab
  accepts connections well before this script finishes, so a user who opens it early
  would otherwise see an empty or half-populated file browser and assume the Launchable
  is broken. `SETUP-IN-PROGRESS.md` is written before any slow work, rewritten by the
  `ERR` trap if provisioning fails, and removed on success.
- The welcome text lives in the Launchable's Source configuration, not a heredoc or
  release tarball. Brev renders it on the Launchable webpage and copies it to
  `$HOME/welcome.md`; keeping the console copy synchronized with this directory is a
  manual deployment step.
- The setup script has a 16 KiB limit. Brev rejects anything larger, which is why
  the script carries short comments pointing here rather than full explanations. Check
  `wc -c script/brev/setup.sh` before pasting.
- The script runs as the instance user, not root, so Python and customer-facing assets
  install under `$HOME`. The sole host-level dependency is `gcc` plus `libc6-dev`:
  Triton JIT-compiles a CUDA driver shim the first time vLLM uses the GPU. Some provider
  images include these packages and others do not, so the script installs them with
  non-interactive passwordless `sudo` when needed. If the provider does not grant that
  permission, setup fails before presenting the notebooks as ready.
- The venv is deliberately separate from Brev's. Installing `[cu129,engine]` into
  `$HOME/.venv` would resolve torch, vllm, and flashinfer alongside jupyterlab's own
  pins. If that breaks the notebook server, the customer's only interface is gone and
  there is no error they can act on. A separate venv keeps a failed install recoverable.
- The instance username varies by provider -- `shadeform` on the Shadeform-brokered
  offerings, likely `ubuntu` on direct ones. The script never names an account; it uses
  `$HOME` and `id -un`.
- The image ships its own `$HOME/.venv` on Python 3.12, which is what the Brev
  Jupyter server runs in. `uv` discovers it by walking up from the working directory, so
  the script pins `VIRTUAL_ENV` in both the kernelspec and `.nss-env.sh` to keep
  `uv pip install` from silently targeting it. Do not delete or upgrade that venv --
  breaking it breaks the notebook server, which is the customer's only interface.
- The kernel is registered as `python3`, not a custom name, because the tutorial
  notebooks declare `kernelspec.name = "python3"`. Any other name would force the
  customer to switch kernels by hand.
- Kernel precedence is counterintuitive. `jupyter_core` puts the server's own
  environment *ahead* of `~/.local/share/jupyter` whenever the server runs inside a
  virtualenv (`prefer_environment_over_user()`), and Brev's does. Registering only at
  user scope is silently shadowed by Brev's 3.12 kernel. The script asks the server's
  interpreter for `jupyter_path("kernels")[0]` and writes there, backing up any existing
  `kernel.json` to `kernel.json.orig`.
- The kernelspec holds secrets, so it stays mode 0600 inside `$HOME`. An earlier
  version also mirrored it to `/usr/local/share/jupyter` at 0644 via sudo, which would
  have put `NSS_INFERENCE_KEY` and `HF_TOKEN` in a world-readable file. Do not
  reintroduce that -- it also could not win precedence over a venv-scoped kernel anyway.
- Most providers do not support stop/start. The instance bills from creation until
  deletion, with no pause. Nebius was the exception at the time of writing.
- Tutorials arrive via tarball, not `git clone`. `--strip-components` lands them flat
  in `tutorials/` rather than the repo's `docs/tutorials/` nesting, and the image is not
  required to have `git`. The tarball is pinned to the tag matching the installed wheel,
  falling back to `main`, so the notebooks can never call APIs newer than the package
  they run against.
- The tar member is an exact path, not a glob. GNU tar does not enable wildcards for
  member names by default, and bsdtar rejects `--wildcards` outright, so no single glob
  form works on both. The archive's top-level directory name also varies by ref
  (`Safe-Synthesizer-0.1.8` for a tag, `Safe-Synthesizer-main` for a branch), so the
  script reads it from the archive rather than guessing.
- Never pipe `tar -t` into `head`. Closing the pipe early makes GNU tar fail with
  `stdout: write error` on SIGPIPE, and `pipefail` turns that into a failed run. bsdtar
  exits silently on SIGPIPE, so this passes on macOS and fails on the instance. The
  script reads the listing into a variable and takes the prefix with `${listing%%/*}`.

### Verification

After deploying, check `$HOME/.nss-setup.log` for the `setup complete` banner, then in a
notebook confirm the environment resolved correctly:

```python
import sys
print(sys.executable)  # expect $HOME/.nss-venv/bin/python
```

That must hold without touching the kernel picker -- the customer should never have
to switch kernels. If it reports a 3.12 path instead, kernel registration landed in a
directory the server does not consult; check the `registered kernel at ...` lines in the
log against `jupyter kernelspec list`.

Then run `tutorials/safe-synthesizer-101.ipynb` end to end -- roughly 15 minutes on
an A100 or H100.

### Tested configurations

| Date | Provider | GPU | Result |
| ---- | -------- | --- | ------ |
| 2026-07-29 | Shadeform-brokered | H100 PCIe 80 GiB | uv install, venv creation, `[cu129,engine]` install, tutorial fetch |

Add a row when you validate another provider or GPU class. The A100-or-larger figure in
the main docs looks conservative for the default 3B model; if a 48 GiB card proves
sufficient, record it here and correct the requirement in `README.md`,
`docs/index.md`, and `docs/user-guide/getting-started.md`.
