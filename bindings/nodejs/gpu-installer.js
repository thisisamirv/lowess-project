'use strict'

// One-time downloader/installer for the opt-in GPU-enabled fastlowess build.
//
// The GPU backend (wgpu) is not included in the native binaries published to
// npm. This module fetches a prebuilt GPU-enabled `.node` addon from the
// matching GitHub Release (built by `.github/workflows/release-gpu.yml`) and
// saves it next to this file, under the platform-specific filename that
// index.js's loader checks before falling back to the optionalDependency
// package.
//
// napi-rs fully regenerates index.js on every `napi build`, so this logic
// lives here (and gets re-attached to index.js's exports by
// scripts/patch-index.js after each build) instead of inside index.js itself.

const path = require('path')
const { execFileSync } = require('child_process')
const { readFileSync, copyFileSync, existsSync } = require('fs')
const gpuDownload = require('./gpu-download')

const { version } = require('./package.json')

// GPU artifacts across all versions live in this one perpetual release
// instead of cluttering each version's own release page; the source version
// is embedded in each asset's filename instead.
const GPU_RELEASE_TAG = 'gpu-builds'

// Musl detection, mirrored from the isMusl() helper napi-rs generates into
// index.js (that copy gets wiped on every `napi build`, so it's duplicated
// here rather than imported).
const isFileMusl = (f) => f.includes('libc.musl-') || f.includes('ld-musl-')

function isMusl() {
    if (process.platform !== 'linux') return false
    try {
        return readFileSync('/usr/bin/ldd', 'utf-8').includes('musl')
    } catch {
        // fall through
    }
    try {
        return execFileSync('ldd', ['--version'], { encoding: 'utf8' }).includes('musl')
    } catch {
        return false
    }
}

// These 9 platforms are built by release-gpu.yml.
function currentPlatformSuffix() {
    if (process.platform === 'win32') {
        if (process.arch === 'x64') return 'win32-x64-msvc'
        if (process.arch === 'arm64') return 'win32-arm64-msvc'
        return null
    }
    if (process.platform === 'darwin') {
        if (process.arch === 'x64') return 'darwin-x64'
        if (process.arch === 'arm64') return 'darwin-arm64'
        return null
    }
    if (process.platform === 'linux') {
        if (process.arch === 'x64') return isMusl() ? 'linux-x64-musl' : 'linux-x64-gnu'
        if (process.arch === 'arm64') return isMusl() ? 'linux-arm64-musl' : 'linux-arm64-gnu'
        if (process.arch === 'arm') return 'linux-arm-gnueabihf'
        return null
    }
    return null
}

// Runs the availability check in a short-lived child process instead of
// `require()`-ing index.js in-process: on Windows, a loaded native addon
// keeps its `.node` file locked for the life of the process, which would
// make the download below fail with EBUSY when overwriting that same file.
function gpuAvailable() {
    try {
        const out = execFileSync(process.execPath, ['-e', "process.stdout.write(String(require('./index.js').gpu_enabled()))"], {
            cwd: __dirname,
            encoding: 'utf8',
        })
        return out.trim() === 'true'
    } catch {
        return false
    }
}

/**
 * Install the GPU-enabled fastlowess native addon for this platform, then
 * restart Node.js to use it.
 *
 * Fetches a prebuilt `.node` addon (built with the `gpu` Cargo feature) from
 * the matching GitHub Release and saves it as `fastlowess.<platform>.node`
 * next to this file — the same local-override path index.js's loader
 * already checks first. A running process cannot swap an already-loaded
 * native addon, so a restart is required afterwards.
 *
 * @param {{ yes?: boolean, localPath?: string }} [options] Pass
 *   `{ yes: true }` to skip the interactive y/N confirmation prompt
 *   (required when stdin is not a TTY). Pass `{ localPath: '/path/to/addon.node' }`
 *   to install an already-built `.node` file instead of downloading one —
 *   useful for testing the installer itself, or installing an unreleased
 *   build.
 */
async function installGpu(options = {}) {
    const { yes = false, localPath } = options
    if (gpuAvailable()) {
        console.log('GPU backend is already active.')
        return
    }

    const suffix = currentPlatformSuffix()
    if (!suffix) {
        throw new Error(
            `No prebuilt GPU binary available for ${process.platform}-${process.arch}. ` +
            'Build from source instead: `npx napi build --release --features gpu`.'
        )
    }

    const destPath = path.join(__dirname, `fastlowess.${suffix}.node`)

    if (localPath !== undefined) {
        if (!existsSync(localPath)) {
            throw new Error(`No such file: ${localPath}`)
        }

        if (!yes) {
            const ok = await gpuDownload.confirm(
                `Install ${localPath} in place of the current build? [y/N] `
            )
            if (!ok) {
                console.log('Aborted.')
                return
            }
        }

        console.log(`Installing ${localPath} ...`)
        copyFileSync(localPath, destPath)
        console.log(`GPU backend installed at ${destPath}.`)
        console.log('Restart Node.js for the change to take effect.')
        return
    }

    const assetName = `fastlowess-gpu-v${version}.${suffix}.node`
    const url = `https://github.com/${gpuDownload.REPO}/releases/download/${GPU_RELEASE_TAG}/${assetName}`

    if (!yes) {
        const ok = await gpuDownload.confirm(
            `Download and install ${assetName} from github.com/${gpuDownload.REPO}? [y/N] `
        )
        if (!ok) {
            console.log('Aborted.')
            return
        }
    }

    console.log(`Downloading ${url} ...`)
    try {
        await gpuDownload.downloadToFile(url, destPath)
    } catch (e) {
        throw new Error(
            `Failed to download ${url}: ${e.message}\n` +
            'A matching GPU build may not exist for this platform/version yet.'
        )
    }

    console.log(`GPU backend installed at ${destPath}.`)
    console.log('Restart Node.js for the change to take effect.')
}

module.exports = { gpuAvailable, installGpu }
