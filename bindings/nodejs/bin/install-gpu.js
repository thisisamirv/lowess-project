#!/usr/bin/env node
'use strict'

require('../gpu-installer.js')
    .installGpu()
    .catch((err) => {
        console.error(err.message || err)
        process.exitCode = 1
    })
