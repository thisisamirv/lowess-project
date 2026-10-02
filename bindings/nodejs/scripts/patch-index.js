'use strict'

// napi-rs regenerates the loader and declarations on every build. Restore the
// GPU sidecar hook, installer export, and async result type after generation.

const fs = require('fs')
const path = require('path')
const { version } = require('../package.json')

const indexPath = path.join(__dirname, '..', 'index.js')
const declarationsPath = path.join(__dirname, '..', 'index.d.ts')
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

let declarations = fs.readFileSync(declarationsPath, 'utf8')
const asyncResultType = /(fit_async\([^\r\n]*\): Promise<)unknown(>)/
declarations = declarations.replace(asyncResultType, '$1LowessResult$2')
if (!/fit_async\([^\r\n]*\): Promise<LowessResult>/.test(declarations)) {
    throw new Error('Could not locate the generated fit_async declaration')
}
fs.writeFileSync(declarationsPath, declarations)
