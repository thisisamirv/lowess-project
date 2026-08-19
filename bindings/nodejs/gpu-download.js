'use strict'

// Small dependency-free helpers for downloading the opt-in GPU-enabled
// native binding from a GitHub Release, with a redirect-following HTTPS GET
// and an interactive y/N confirmation prompt.

const https = require('https')
const fs = require('fs')
const path = require('path')
const readline = require('readline/promises')

const REPO = 'thisisamirv/lowess-project'

function get(url, redirectsLeft = 5) {
    return new Promise((resolve, reject) => {
        https
            .get(url, { headers: { 'User-Agent': 'fastlowess-install-gpu' } }, (res) => {
                const { statusCode, headers } = res
                if ([301, 302, 303, 307, 308].includes(statusCode) && headers.location) {
                    res.resume()
                    if (redirectsLeft <= 0) {
                        reject(new Error('Too many redirects'))
                        return
                    }
                    resolve(get(headers.location, redirectsLeft - 1))
                    return
                }
                resolve(res)
            })
            .on('error', reject)
    })
}

async function downloadToFile(url, destPath) {
    const res = await get(url)
    if (res.statusCode !== 200) {
        res.resume()
        throw new Error(`Download failed: HTTP ${res.statusCode} for ${url}`)
    }
    await fs.promises.mkdir(path.dirname(destPath), { recursive: true })
    await new Promise((resolve, reject) => {
        const file = fs.createWriteStream(destPath)
        res.pipe(file)
        file.on('finish', () => file.close(resolve))
        file.on('error', reject)
        res.on('error', reject)
    })
}

async function confirm(message) {
    if (!process.stdin.isTTY) {
        throw new Error(
            'install_gpu() requires confirmation. Pass { yes: true } to proceed non-interactively.'
        )
    }
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout })
    try {
        const answer = await rl.question(message)
        return /^y(es)?$/i.test(answer.trim())
    } finally {
        rl.close()
    }
}

module.exports = { REPO, downloadToFile, confirm }
