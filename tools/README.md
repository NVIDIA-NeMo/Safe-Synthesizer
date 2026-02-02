# developer tools and setup

This folder holds setup scripts for binary tools and meta-configuration information needed
to develop with this repo. Services can keep tools for the service installed elsewhere, but
general purpose tools should be configured here.

the `bootstrap_tools.sh` file will scan this folder for installation scripts and run them all. from the root of the repo, run `make bootstrap-tools` to install them all.

## Guidelines

the `defs.sh` file defines common environment variables that can be sourced from other scripts throughout the repo, but primarily in this folder. The most important variable that you can configure in your own `.[zsh|bash]rc` is `AIRE_REPOS`, which should be set to the path where you clone AIRE-related git repos. It will default to `$HOME/dev` (see [this link](https://nvidia.sharepoint.com/sites/EndpointProtection/SitePages/Home.aspx) for an explanation).

### adding a tool install script

- make a script, `install_<name>.sh` or `bootstrap_<name>.sh`. use the former for scripts that just download a simple binary and require minimal other changes, use the latter for tools that have other deps or require more followup from the user.
- pin the version that you want to install / make it explicit
- ensure your installation works on macos (apple silicon / arm64) and debian linux amd64
- add new binary tools to either:
    1. direnv-managed repo-local path
    2. `$HOME/.local/bin` - the XDG default for linux; we'll use this on macos too
    3. package-manager default with standard PATH - e.g., homebrew/apt/etc.
- print out or otherwise document how to add to the appropriate startup rc file, e.g.:
  - create or symlink a named `.config|.rc` file in `XDG_CONFIG_HOME` (`$HOME/.config/`) and print how the user can source that file

## Testing

There is a small docker script you can run to test installation on linux:

```
bash test_tool_install.sh
```

check to make sure your tool installs correctly.  `az` and `kubectl` are known failures here - az should be installed via uv pip on linux as the offical script doesn't like containerized python.
