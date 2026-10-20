[CmdletBinding()]
param(
    [switch] $Yes,
    [string] $Msys2Root = $(if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { 'C:\dev\msys64' })
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $root 'target'
$dist = Join-Path $target 'reims-vgpu-windows-dist'
$dataRoot = Join-Path $env:LOCALAPPDATA 'reims-ctl'
$qemuRoot = Join-Path $dataRoot 'qemu\current'
$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
$cargoBin = Join-Path $cargoHome 'bin'
$bash = Join-Path $Msys2Root 'usr\bin\bash.exe'

function Stop-Install([string] $Message) { throw "install-ctl: $Message" }
function Invoke-Checked([string] $Program, [string[]] $Arguments) {
    Write-Host "> $Program $($Arguments -join ' ')"
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { Stop-Install "$Program failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Stop-Install 'Git for Windows is required.' }
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargo -and (Test-Path (Join-Path $cargoBin 'cargo.exe'))) { $cargo = Get-Item (Join-Path $cargoBin 'cargo.exe') }
if (-not $cargo) { Stop-Install 'Rust is required. Install it from https://rustup.rs, open a new PowerShell window, then run this again.' }
$cargoExe = if ($cargo.Source) { $cargo.Source } else { $cargo.Path }
if (-not (Test-Path $bash)) { Stop-Install "MSYS2 is required at '$Msys2Root'. Install MSYS2 UCRT64 or pass -Msys2Root." }
$env:PATH = "$(Join-Path $Msys2Root 'ucrt64\bin');$(Join-Path $Msys2Root 'usr\bin');$env:PATH"

$rustup = Get-Command rustup -ErrorAction SilentlyContinue
if ($rustup) { Invoke-Checked $rustup.Source @('target', 'add', 'x86_64-pc-windows-gnu') }

$packages = @(
    'mingw-w64-ucrt-x86_64-gcc', 'mingw-w64-ucrt-x86_64-ninja',
    'mingw-w64-ucrt-x86_64-meson', 'mingw-w64-ucrt-x86_64-pkgconf',
    'mingw-w64-ucrt-x86_64-glib2', 'mingw-w64-ucrt-x86_64-pixman',
    'mingw-w64-ucrt-x86_64-libslirp', 'mingw-w64-ucrt-x86_64-libusb',
    'mingw-w64-ucrt-x86_64-clang', 'mingw-w64-ucrt-x86_64-lld',
    'mingw-w64-ucrt-x86_64-zlib', 'make', 'python', 'flex', 'bison'
)
$missing = & $bash -c 'for p; do pacman -Q "$p" >/dev/null 2>&1 || printf "%s\n" "$p"; done' -- @packages
if ($LASTEXITCODE -ne 0) { Stop-Install 'Could not query MSYS2 packages. Open the MSYS2 UCRT64 shell and update its package database.' }
$missing = @($missing | Where-Object { $_ -and $_.Trim() })
if ($missing.Count -gt 0) {
    Write-Host "MSYS2 packages to install: $($missing -join ', ')"
    if (-not $Yes -and (Read-Host 'Install these packages with pacman? [y/N]') -notmatch '^(y|yes)$') {
        Stop-Install 'Install the listed MSYS2 packages, then run this script again.'
    }
    Invoke-Checked $bash (@('-c', 'export PATH=/ucrt64/bin:/usr/bin:$PATH; pacman -S --needed --noconfirm "$@"', 'bash') + $missing)
}

$env:PATH = "$(Join-Path $Msys2Root 'ucrt64\bin');$(Join-Path $Msys2Root 'usr\bin');$env:PATH"

Invoke-Checked $cargoExe @('build', '--locked', '--release', '--target', 'x86_64-pc-windows-gnu', '-p', 'reims-vgpu-c', '--lib', '--features', 'reims-vgpu-c/vulkan')
$lib = Join-Path $target 'x86_64-pc-windows-gnu\release\libreims_vgpu_c.a'
if (-not (Test-Path $lib)) { Stop-Install "The Windows static library was not built: $lib" }

$submodule = Join-Path $root 'vendor\qemu'
if (Test-Path (Join-Path $submodule '.git')) {
    $subStatus = & git -C $submodule status --porcelain
    if ($LASTEXITCODE -ne 0) { Stop-Install 'Could not inspect vendor/qemu.' }
    if ($subStatus) { Stop-Install 'vendor/qemu has local changes. Commit or preserve them before the installer updates its configured branch.' }
}
Invoke-Checked 'git' @('-C', $root, 'submodule', 'update', '--init', '--remote', '--depth', '1', 'vendor/qemu')

New-Item -ItemType Directory -Force -Path (Join-Path $dist 'lib\pkgconfig'), (Join-Path $dist 'include'), (Join-Path $dist 'share\reims-vgpu') | Out-Null
Copy-Item -Force $lib (Join-Path $dist 'lib\libreims_vgpu.a')
Copy-Item -Force (Join-Path $root 'crates\reims-vgpu-c\include\reims_vgpu.h') (Join-Path $dist 'include\reims_vgpu.h')
$libHash = (Get-FileHash $lib -Algorithm SHA256).Hash.Substring(0, 10).ToLowerInvariant()
$headerHash = (Get-FileHash (Join-Path $dist 'include\reims_vgpu.h') -Algorithm SHA256).Hash.Substring(0, 10).ToLowerInvariant()
$qemuCommit = (& git -C $submodule rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { Stop-Install 'Could not read vendor/qemu revision.' }
$buildName = "reims-qemu-$($qemuCommit.Substring(0, 10))-$libHash-$headerHash"
$qemuBuild = Join-Path $target $buildName
$stageName = "reims-qemu-stage-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
$qemuStage = Join-Path $target $stageName
$romBuild = Join-Path $target 'reims-firmware-windows'

$cygpath = Join-Path $Msys2Root 'usr\bin\cygpath.exe'
$env:REIMS_ROOT_POSIX = (& $cygpath -u $root).Trim()
$env:REIMS_TARGET_POSIX = (& $cygpath -u $target).Trim()
$env:REIMS_DIST_POSIX = (& $cygpath -u $dist).Trim()
$env:REIMS_QEMU_BUILD_POSIX = (& $cygpath -u $qemuBuild).Trim()
$env:REIMS_QEMU_STAGE_POSIX = (& $cygpath -u $qemuStage).Trim()
$env:REIMS_ROM_BUILD_POSIX = (& $cygpath -u $romBuild).Trim()
$buildScript = Join-Path $target "$stageName.sh"
@'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/ucrt64/bin:/usr/bin
ROOT="$REIMS_ROOT_POSIX"
DIST="$REIMS_DIST_POSIX"
BUILD="$REIMS_QEMU_BUILD_POSIX"
STAGE="$REIMS_QEMU_STAGE_POSIX"
ROM_BUILD="$REIMS_ROM_BUILD_POSIX"
mkdir -p "$DIST/lib/pkgconfig" "$DIST/include" "$DIST/share/reims-vgpu" "$BUILD" "$ROM_BUILD"
make -C "$ROOT/firmware" BUILD="$ROM_BUILD" rom
cp "$ROM_BUILD/ReimsVgpu.rom" "$DIST/share/reims-vgpu/ReimsVgpu.rom"
cat > "$DIST/lib/pkgconfig/reims-vgpu.pc" <<'PC'
prefix=${pcfiledir}/../..
libdir=${prefix}/lib
includedir=${prefix}/include
optionrom=${prefix}/share/reims-vgpu/ReimsVgpu.rom
Name: reims-vgpu
Description: Reims macOS virtual GPU
Version: 0.1.0
Libs: -L${libdir} -lreims_vgpu
Libs.private: -lkernel32 -lntdll -luserenv -lws2_32 -ldbghelp -lonecore -lshcore -lmincore
Cflags: -I${includedir}
PC
export PKG_CONFIG_PATH="$DIST/lib/pkgconfig:/ucrt64/lib/pkgconfig:/usr/lib/pkgconfig"
mkdir -p "$BUILD"
cd "$BUILD"
MSYS2_ARG_CONV_EXCL='--prefix=;-Dprefix=;-Dbindir=;-Ddatadir=' "$ROOT/vendor/qemu/configure" --prefix=/reims-ctl-qemu --bindir=bin --datadir=share/qemu --target-list=x86_64-softmmu \
  --enable-whpx --enable-pvg --disable-fdt --disable-docs --disable-plugins --disable-tests --disable-werror
ninja -C "$BUILD" qemu-system-x86_64.exe qemu-img.exe
mkdir -p "$STAGE"
DESTDIR="$STAGE" ninja -C "$BUILD" install
BIN="$STAGE/reims-ctl-qemu/bin"
for exe in "$BIN/qemu-system-x86_64.exe" "$BIN/qemu-img.exe"; do
  ldd "$exe" | awk '$3 ~ /^\/ucrt64\/bin\// || $3 ~ /^\/usr\/bin\// {print $3}'
done | sort -u | while IFS= read -r dll; do
  [ -f "$dll" ] && cp -n "$dll" "$BIN/"
done
'@ | Set-Content -LiteralPath $buildScript -Encoding ascii
Invoke-Checked $bash @('--noprofile', '--norc', $buildScript)

$installedPrefix = Join-Path $qemuStage 'reims-ctl-qemu'
$installedBin = Join-Path $installedPrefix 'bin'
foreach ($file in @('qemu-system-x86_64.exe', 'qemu-img.exe')) {
    if (-not (Test-Path (Join-Path $installedBin $file))) { Stop-Install "QEMU install is missing bin\$file" }
}
$qemuData = Join-Path $installedPrefix 'share\qemu'
if (-not (Test-Path (Join-Path $qemuData 'ReimsVgpu.rom'))) { Stop-Install 'QEMU install is missing its Reims option ROM.' }
$runningQemu = Get-CimInstance Win32_Process -Filter "Name='qemu-system-x86_64.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($qemuRoot, [StringComparison]::OrdinalIgnoreCase) }
if ($runningQemu) { Stop-Install 'QEMU is running from the install directory. Stop it, then run the installer again.' }
New-Item -ItemType Directory -Force -Path $qemuRoot | Out-Null
Copy-Item -Path (Join-Path $installedPrefix '*') -Destination $qemuRoot -Recurse -Force

Invoke-Checked $cargoExe @('install', '--locked', '--force', '--path', (Join-Path $root 'crates\reims-ctl'), '--root', $cargoHome)
$ctl = Join-Path $cargoBin 'reims-ctl.exe'
if (-not (Test-Path $ctl)) { Stop-Install "cargo install left no $ctl" }
$commit = (& git -C $root rev-parse HEAD).Trim()
Invoke-Checked $ctl @('__record-install', '--repo', $root, '--qemu', $qemuRoot, '--commit', $commit)
$env:PATH = "$cargoBin;$env:PATH"
Write-Host ''
Write-Host 'reims-ctl and its Windows QEMU are installed.'
Write-Host "  QEMU: $qemuRoot"
Write-Host "  Run:  $ctl doctor"
& $ctl doctor
if ($LASTEXITCODE -ne 0) { Stop-Install 'reims-ctl doctor found a problem; see the report above.' }
