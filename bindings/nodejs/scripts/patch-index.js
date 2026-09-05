'use strict'

// napi-rs fully regenerates index.js on every `napi build`, wiping any manual
// additions. Re-append the `installGpu` export after each build so
// `require('fastlowess').installGpu()` (the documented API) keeps working.

const fs = require('fs')
const path = require('path')

const indexPath = path.join(__dirname, '..', 'index.js')
const marker = "module.exports.installGpu = require('./gpu-installer.js').installGpu"

let contents = fs.readFileSync(indexPath, 'utf8')
if (!contents.includes(marker)) {
    contents += `\n${marker}\n`
    fs.writeFileSync(indexPath, contents)
}
