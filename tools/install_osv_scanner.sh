#!/usr/bin/env bash
set -eux
REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

# Install osv-scanner for license checking and vulnerability scanning
install_osv_scanner() {
    if command -v osv-scanner >/dev/null; then
        echo "osv-scanner is already installed at $(which osv-scanner)"
        osv-scanner --version
        exit 0
    fi

    version="v2.2.3"
    echo "Installing osv-scanner ${version}"

    # Determine the binary name based on OS and architecture
    if [ "$OS" == "linux" ]; then
        if [ "$ARCH" == "amd64" ]; then
            binary_name="osv-scanner_linux_amd64"
        elif [ "$ARCH" == "arm64" ]; then
            binary_name="osv-scanner_linux_arm64"
        else
            echo "Error: Unsupported architecture: $ARCH"
            exit 1
        fi
    elif [ "$OS" == "darwin" ]; then
        if [ "$ARCH" == "amd64" ]; then
            binary_name="osv-scanner_darwin_amd64"
        elif [ "$ARCH" == "arm64" ]; then
            binary_name="osv-scanner_darwin_arm64"
        else
            echo "Error: Unsupported architecture: $ARCH"
            exit 1
        fi
    else
        echo "Error: Unsupported OS: $OS"
        exit 1
    fi

    # Download the binary
    echo "Downloading ${binary_name}..."
    curl -L "https://github.com/google/osv-scanner/releases/download/${version}/${binary_name}" -o osv-scanner

    # Make it executable
    chmod +x osv-scanner

    # Install to user's local bin directory
    mkdir -p "$HOME/.local/bin"
    mv osv-scanner "$HOME/.local/bin/osv-scanner"

    echo "osv-scanner installed successfully to $HOME/.local/bin/osv-scanner"
    echo "Make sure $HOME/.local/bin is in your PATH"

    # Verify installation
    if command -v osv-scanner >/dev/null; then
        osv-scanner --version
    else
        echo "Warning: osv-scanner installed but not found in PATH"
        echo "Add the following to your shell config file (e.g., ~/.bashrc or ~/.zshrc):"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
}

install_osv_scanner
