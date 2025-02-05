#!/data/data/com.termux/files/usr/bin/bash

# Set strict error handling
set -euo pipefail

# Configuration variables (modified for clone)
UBUNTU_VERSION='24.10'
INSTALL_DIR="ubuntu-fs-clone"
BIND_DIR="ubuntu-binds-clone"
START_SCRIPT="startubuntu-clone.sh"
DNS_SERVERS=("1.1.1.1" "8.8.8.8" "9.9.9.9")

# Color codes for pretty printing
readonly RED="\x1b[38;5;203m"
readonly YELLOW="\x1b[38;5;214m"
readonly GREEN="\x1b[38;5;83m"
readonly BLUE="\x1b[38;5;87m"
readonly PURPLE="\x1b[38;5;127m"
readonly RESET="\e[0m"

# Get current time
time1="$( date +"%r" )"

# Logger function
log() {
    local level=$1
    local message=$2
    printf "${YELLOW}[${time1}]${RESET} ${level} ${BLUE}${message}${RESET}\n"
}

# Error handler
error_handler() {
    log "${RED}[ERROR]:" "An error occurred on line $1"
    exit 1
}

trap 'error_handler ${LINENO}' ERR

# Check system requirements
check_requirements() {
    local missing_deps=()
    
    for cmd in proot wget tar; do
        if ! command -v "$cmd" >/dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log "${RED}[ERROR]:" "Missing required packages: ${missing_deps[*]}"
        log "${GREEN}[INFO]:" "Please install them using: pkg install ${missing_deps[*]}"
        exit 1
    fi
}

# Detect architecture
get_architecture() {
    local arch=$(dpkg --print-architecture)
    case "$arch" in
        aarch64) echo "arm64";;
        arm) echo "armhf";;
        amd64|x86_64) echo "amd64";;
        *) 
            log "${RED}[ERROR]:" "Unsupported architecture: $arch"
            exit 1
        ;;
    esac
}

# Download Ubuntu rootfs
download_rootfs() {
    local architecture=$(get_architecture)
    local url="https://cdimage.ubuntu.com/ubuntu-base/releases/${UBUNTU_VERSION}/release/ubuntu-base-${UBUNTU_VERSION}-base-${architecture}.tar.gz"
    
    log "${GREEN}[INFO]:" "Downloading Ubuntu ${UBUNTU_VERSION} (${architecture})..."
    wget "$url" -q --show-progress -O ubuntu-clone.tar.gz || {
        log "${RED}[ERROR]:" "Failed to download Ubuntu rootfs"
        exit 1
    }
}

# Extract and setup Ubuntu
setup_ubuntu() {
    local cur_dir=$(pwd)
    
    # Create necessary directories
    mkdir -p "$INSTALL_DIR"
    
    # Extract rootfs
    log "${GREEN}[INFO]:" "Extracting Ubuntu rootfs..."
    proot --link2symlink tar -zxf ubuntu-clone.tar.gz -C "$INSTALL_DIR" --exclude='dev' || {
        log "${RED}[ERROR]:" "Failed to extract rootfs"
        exit 1
    }
    
    # Configure DNS
    log "${GREEN}[INFO]:" "Configuring DNS..."
    printf "%s\n" "${DNS_SERVERS[@]/#/nameserver }" > "$INSTALL_DIR/etc/resolv.conf"
    
    # Setup basic system files
    log "${GREEN}[INFO]:" "Setting up system files..."
    echo -e "#!/bin/sh\nexit" > "$INSTALL_DIR/usr/bin/groups"
    chmod +x "$INSTALL_DIR/usr/bin/groups"
}

# Create startup script
create_startup_script() {
    log "${GREEN}[INFO]:" "Creating startup script..."
    
    cat > "$START_SCRIPT" <<- 'EOF'
#!/bin/bash
cd $(dirname $0)
unset LD_PRELOAD
command="proot"
command+=" --link2symlink"
command+=" -0"
command+=" -r ubuntu-fs-clone"
command+=" -b /dev"
command+=" -b /proc"
command+=" -b /sys"
command+=" -b ubuntu-fs-clone/tmp:/dev/shm"
command+=" -b /data/data/com.termux"
command+=" -b /:/host-rootfs"
command+=" -b /sdcard"
command+=" -b /storage"
command+=" -b /mnt"
command+=" -w /root"
command+=" /usr/bin/env -i"
command+=" HOME=/root"
command+=" PATH=/usr/local/sbin:/usr/local/bin:/bin:/usr/bin:/sbin:/usr/sbin:/usr/games:/usr/local/games"
command+=" TERM=$TERM"
command+=" LANG=C.UTF-8"
command+=" /bin/bash --login"

# Process custom bind mounts
if [ -d "ubuntu-binds-clone" ] && [ -n "$(ls -A ubuntu-binds-clone 2>/dev/null)" ]; then
    for f in ubuntu-binds-clone/*; do
        [ -f "$f" ] && . "$f"
    done
fi

# Execute command or start shell
if [ $# -gt 0 ]; then
    $command -c "$*"
else
    exec $command
fi
EOF

    # Fix permissions
    termux-fix-shebang "$START_SCRIPT"
    chmod +x "$START_SCRIPT"
}

# Main installation function
install_ubuntu() {
    # Check if already installed
    if [ -d "$INSTALL_DIR" ]; then
        log "${YELLOW}[WARNING]:" "Ubuntu clone is already installed. Skipping installation."
        return 0
    fi
    
    # Create bind directory
    mkdir -p "$BIND_DIR"
    
    # Perform installation steps
    check_requirements
    download_rootfs
    setup_ubuntu
    create_startup_script
    
    # Cleanup
    log "${GREEN}[INFO]:" "Cleaning up..."
    rm -f ubuntu-clone.tar.gz
    
    log "${GREEN}[INFO]:" "Ubuntu clone installation completed successfully!"
    log "${GREEN}[INFO]:" "Start Ubuntu clone by running: ./${START_SCRIPT}"
}

# Main script execution
main() {
    if [ "${1:-}" = "-y" ]; then
        install_ubuntu
    else
        printf "${YELLOW}[${time1}]${RESET} ${PURPLE}[QUESTION]:${RESET} ${BLUE}Do you want to install Ubuntu clone in Termux? [Y/n] ${RESET}"
        read -r response
        case "$response" in
            [Yy]*|"") install_ubuntu ;;
            *) log "${RED}[ERROR]:" "Installation aborted." ;;
        esac
    fi
}

main "$@"
