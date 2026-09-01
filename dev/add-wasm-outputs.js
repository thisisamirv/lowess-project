'use strict';

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Accept binding dir as first argument; fall back to cwd (e.g. when invoked via npm run snippets)
const BINDING_DIR = process.argv[2] ? path.resolve(process.argv[2]) : process.cwd();
const DOCS_DIR = path.join(BINDING_DIR, 'src', 'content', 'docs');
const PKG_NAME = JSON.parse(fs.readFileSync(path.join(BINDING_DIR, 'package.json'), 'utf-8')).name;
// wasm-pack outputs to pkg/; read main entry from there
const pkgMeta = JSON.parse(fs.readFileSync(path.join(BINDING_DIR, 'pkg', 'package.json'), 'utf-8'));
const INDEX_PATH = path.join(BINDING_DIR, 'pkg', pkgMeta.main).replace(/\\/g, '/');

const SKIP_FILES = new Set();
const SKIP_PATTERNS = [];

const BT = '```';
const REQUIRE_RE = new RegExp(`require\\(['"]${PKG_NAME}['"]\\)`, 'g');

function runSnippet(code) {
    const patched = code.replace(REQUIRE_RE, `require('${INDEX_PATH}')`);
    const tmpDir = os.tmpdir();
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
    const tmp = path.join(tmpDir, `snippet-${Date.now()}-${Math.random().toString(36).slice(2)}.js`);
    fs.writeFileSync(tmp, patched, 'utf-8');
    try {
        const output = execFileSync(process.execPath, [tmp], {
            cwd: BINDING_DIR,
            timeout: 60000,
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'pipe'],
        }).trim() || null;
        return { output, error: null };
    } catch (e) {
        return { output: null, error: e.stderr?.trim() || e.message };
    } finally {
        try { fs.unlinkSync(tmp); } catch { /* ignore */ }
    }
}

function processFile(filepath) {
    if (SKIP_FILES.has(path.basename(filepath))) return { changed: false, errors: [] };

    const original = fs.readFileSync(filepath, 'utf-8').replace(/\r\n/g, '\n');
    let result = '';
    let pos = 0;
    const errors = [];
    const re = /```javascript\n([\s\S]*?)```/g;
    let m;

    while ((m = re.exec(original)) !== null) {
        result += original.slice(pos, m.index + m[0].length);
        pos = m.index + m[0].length;

        const existing = original.slice(pos).match(/^\n\n```output\n[\s\S]*?```/);
        if (existing) pos += existing[0].length;

        if (!SKIP_PATTERNS.some(p => m[1].includes(p)) && !/^import\b/m.test(m[1])) {
            const { output, error } = runSnippet(m[1]);
            if (error) {
                errors.push(`  FAIL ${path.relative(DOCS_DIR, filepath)}\n${error}`);
                if (existing) result += existing[0];
            } else if (output) {
                result += `\n\n${BT}output\n${output}\n${BT}`;
            }
        } else if (existing) {
            result += existing[0];
        }
    }

    result += original.slice(pos);

    const changed = result !== original;
    if (changed) fs.writeFileSync(filepath, result, 'utf-8');
    return { changed, errors };
}

const files = fs.readdirSync(DOCS_DIR)
    .filter(f => f.endsWith('.md') && f !== 'index.md')
    .sort();

let updated = 0;
const allErrors = [];
for (const f of files) {
    process.stdout.write(`  ${f}...`);
    const { changed, errors } = processFile(path.join(DOCS_DIR, f));
    process.stdout.write(changed ? ' updated\n' : '\n');
    if (changed) updated++;
    allErrors.push(...errors);
}
const unchanged = files.length - updated;
console.log(
    `\nDone -- ${files.length} file(s) assessed, ${updated} updated, ${unchanged} already up to date.`
);

if (allErrors.length > 0) {
    process.stderr.write('\nFailed snippets:\n');
    for (const msg of allErrors) process.stderr.write(msg + '\n\n');
    process.exit(1);
}
