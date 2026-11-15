'use strict'

// napi-rs regenerates the loader and declarations on every build. Restore the
// GPU sidecar hook, installer export/types, async result type, status union,
// and Symbol.dispose hooks after generation.

const fs = require('fs')
const path = require('path')
const { version } = require('../package.json')

const indexPath = path.join(__dirname, '..', 'index.js')
const declarationsPath = path.join(__dirname, '..', 'index.d.ts')
const nativeLoaderMarker = 'function requireNative() {'
const gpuLoaderMarker = '// fastlowess versioned GPU override'
const disposeMarker = '// fastlowess Symbol.dispose hooks'
const installGpuExport = "module.exports.installGpu = require('./gpu-installer.js').installGpu"

const gpuLoader = `  ${gpuLoaderMarker}
  if (!process.env.NAPI_RS_NATIVE_LIBRARY_PATH) {
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

const disposeHooks = `
${disposeMarker}
function __fastlowessAttachDispose(ctor) {
  if (!ctor || !ctor.prototype || typeof ctor.prototype.close !== 'function') {
    return
  }
  if (typeof Symbol !== 'undefined' && Symbol.dispose && !ctor.prototype[Symbol.dispose]) {
    ctor.prototype[Symbol.dispose] = function () {
      this.close()
    }
  }
}
__fastlowessAttachDispose(module.exports.OnlineLowess)
__fastlowessAttachDispose(module.exports.StreamingLowess)
__fastlowessAttachDispose(module.exports.LowessResult)
`

let contents = fs.readFileSync(indexPath, 'utf8').replace(/\r\n/g, '\n')
contents = contents.replace(
  /bindingPackageVersion !== '\d+\.\d+\.\d+'/g,
  "bindingPackageVersion !== require('./package.json').version"
)
contents = contents.replace(
  /expected \d+\.\d+\.\d+ but got/g,
  () => "expected ${require('./package.json').version} but got"
)
if (!contents.includes(gpuLoaderMarker)) {
  if (!contents.includes(nativeLoaderMarker)) {
    throw new Error('Could not locate the N-API native loader insertion point')
  }
  contents = contents.replace(
    nativeLoaderMarker,
    `${nativeLoaderMarker}\n${gpuLoader}`
  )
}
if (!contents.includes(installGpuExport)) {
  contents += `\n${installGpuExport}\n`
}
if (!contents.includes(disposeMarker)) {
  contents += disposeHooks
}
fs.writeFileSync(indexPath, contents)

let declarations = fs.readFileSync(declarationsPath, 'utf8').replace(/\r\n/g, '\n')
const asyncResultType = /(fit_async\([\s\S]*?\): Promise<)unknown(>)/
declarations = declarations.replace(asyncResultType, '$1LowessResult$2')
if (!/fit_async\([\s\S]*?\): Promise<LowessResult>/.test(declarations)) {
  throw new Error('Could not locate the generated fit_async declaration')
}
declarations = declarations.replace(
  /(export interface AddPointResult \{\n\s+\/\*\*[\s\S]*?\*\/\n\s+status: )string(\n)/,
  '$1\'filling\' | \'ready\' | \'dropped\'$2'
)
if (!declarations.includes("status: 'filling' | 'ready' | 'dropped'")) {
  // Fallback if napi omits the doc comment block.
  declarations = declarations.replace(
    /(export interface AddPointResult \{\n\s+status: )string(\n)/,
    '$1\'filling\' | \'ready\' | \'dropped\'$2'
  )
}
const installGpuType = `
/** Download or install a prebuilt GPU-enabled native addon sidecar. */
export declare function installGpu(options?: { yes?: boolean; localPath?: string }): Promise<void>
`
if (!declarations.includes('installGpu')) {
  declarations += installGpuType
}
fs.writeFileSync(declarationsPath, declarations)
