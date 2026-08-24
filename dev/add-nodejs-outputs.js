'use strict';

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Accept binding dir as first argument; fall back to cwd (e.g. when invoked via npm run snippets)
const BINDING_DIR = process.argv[2] ? path.resolve(process.argv[2]) : process.cwd();
const DOCS_DIR = path.join(BINDING_DIR, 'docs');
const PKG_NAME = JSON.parse(fs.readFileSync(path.join(BINDING_DIR, 'package.json'), 'utf-8')).name;
// Forward-slash path works on Windows too and avoids escape headaches in the injected require()
const INDEX_PATH = path.join(BINDING_DIR, 'index.js').replace(/\\/g, '/');

// Skip these files entirely (GPU-only or non-runnable)
const SKIP_FILES = new Set(['gpu-backend.md']);
// Skip individual blocks containing any of these strings
const SKIP_PATTERNS = ['installGpu', '"gpu"'];

const BT = '```';
const REQUIRE_RE = new RegExp(`require\\(['"]${PKG_NAME}['"]\\)`, 'g');

function runSnippet(code) {
    const patched = code.replace(REQUIRE_RE, `require('${INDEX_PATH}')`);
    const tmp = path.join(os.tmpdir(), `snippet-${Date.now()}-${Math.random().toString(36).slice(2)}.js`);
    fs.writeFileSync(tmp, patched, 'utf-8');
    try {
        return execFileSync(process.execPath, [tmp], {
            cwd: BINDING_DIR,
            timeout: 15000,
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'ignore'],
        }).trim() || null;
    } catch {
        // Silently skip blocks that fail (partial examples, undefined variables, etc.)
        return null;
    } finally {
        try { fs.unlinkSync(tmp); } catch { /* ignore */ }
    }
}

function processFile(filepath) {
    if (SKIP_FILES.has(path.basename(filepath))) return false;

    const original = fs.readFileSync(filepath, 'utf-8').replace(/\r\n/g, '\n');
    let result = '';
    let pos = 0;
    const re = /```javascript\n([\s\S]*?)```/g;
    let m;

    while ((m = re.exec(original)) !== null) {
        result += original.slice(pos, m.index + m[0].length);
        pos = m.index + m[0].length;

        // Consume any existing runner-output block that directly follows
        const existing = original.slice(pos).match(/^\n\n```output\n[\s\S]*?```/);
        if (existing) pos += existing[0].length;

        if (!SKIP_PATTERNS.some(p => m[1].includes(p))) {
            const out = runSnippet(m[1]);
            if (out) result += `\n\n${BT}output\n${out}\n${BT}`;
        }
    }

    result += original.slice(pos);

    if (result !== original) {
        fs.writeFileSync(filepath, result, 'utf-8');
        return true;
    }
    return false;
}

const files = fs.readdirSync(DOCS_DIR)
    .filter(f => f.endsWith('.md') && f !== 'index.md')
    .sort();

let updated = 0;
for (const f of files) {
    process.stdout.write(`  ${f}...`);
    const changed = processFile(path.join(DOCS_DIR, f));
    process.stdout.write(changed ? ' updated\n' : '\n');
    if (changed) updated++;
}
console.log(`add-nodejs-outputs: ${updated}/${files.length} file(s) updated`);
