'use strict'

// Keep package-lock.json optional platform packages in sync with package.json
// and bindings/nodejs/npm/*/package.json. Needed before those versions are
// published to the registry; bare `npm update` otherwise drops them.

const fs = require('fs')
const path = require('path')

const root = path.join(__dirname, '..')
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'))
const lockPath = path.join(root, 'package-lock.json')
const lock = JSON.parse(fs.readFileSync(lockPath, 'utf8'))

const optional = pkg.optionalDependencies || {}
let changed = 0

for (const [name, range] of Object.entries(optional)) {
  const platform = name.replace(/^fastlowess-/, '')
  const localPath = path.join(root, 'npm', platform, 'package.json')
  if (!fs.existsSync(localPath)) {
    console.warn(`skip missing local package: ${localPath}`)
    continue
  }
  const local = JSON.parse(fs.readFileSync(localPath, 'utf8'))
  const key = `node_modules/${name}`
  const prev = lock.packages[key] || {}
  const entry = {
    version: local.version,
    resolved: `https://registry.npmjs.org/${name}/-/${name}-${local.version}.tgz`,
    optional: true,
    license: local.license,
    engines: local.engines,
  }
  if (local.cpu) entry.cpu = local.cpu
  if (local.os) entry.os = local.os
  if (local.libc) entry.libc = local.libc
  if (prev.integrity) entry.integrity = prev.integrity

  lock.packages[key] = entry
  changed += 1
}

if (lock.packages['']) {
  lock.packages[''].optionalDependencies = { ...optional }
  if (pkg.version) {
    lock.version = pkg.version
    lock.packages[''].version = pkg.version
  }
}

fs.writeFileSync(lockPath, `${JSON.stringify(lock, null, 4)}\n`)
console.log(`Synced ${changed} optional platform package(s) into package-lock.json`)
