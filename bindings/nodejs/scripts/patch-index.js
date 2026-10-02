'use strict'

// napi-rs fully regenerates index.js on every `napi build`, wiping any manual
// additions. Re-append the `installGpu` export after each build so
// `require('fastlowess').installGpu()` (the documented API) keeps working.

const fs = require('fs')
const path = require('path')
const { version } = require('../package.json')

const indexPath = path.join(__dirname, '..', 'index.js')
const nativeLoaderMarker = 'function requireNative() {\n'
const gpuLoaderMarker = '  // fastlowess versioned GPU override\n'
const gpuLoader = `${gpuLoaderMarker}  if (!process.env.NAPI_RS_NATIVE_LIBRARY_PATH) {
        try {
            const gpuBinding = require('./fastlowess.gpu-v${version}.node')
            if (typeof gpuBinding.gpu_enabled === 'function' && gpuBinding.gpu_enabled()) {
                return gpuBinding
            }
        } catch (e) {
            if (e.code !== 'MODULE_NOT_FOUND') loadErrors.push(e)
        }
    }
`
const marker = "module.exports.installGpu = require('./gpu-installer.js').installGpu"

let contents = fs.readFileSync(indexPath, 'utf8')
if (!contents.includes(gpuLoaderMarker)) {
    if (!contents.includes(nativeLoaderMarker)) {
        throw new Error('Could not locate the N-API native loader insertion point')
    }
    contents = contents.replace(nativeLoaderMarker, `${nativeLoaderMarker}${gpuLoader}`)
}
if (!contents.includes(marker)) {
    contents += `\n${marker}\n`
}
fs.writeFileSync(indexPath, contents)
