#!/usr/bin/env node
// TypeDoc preserves the original TypeScript symbol casing (e.g. Lowess.md),
// but Astro/Starlight always lowercases content-collection route slugs, so
// its built-in relative-link resolution (which strips ".md" but keeps the
// casing as authored) points at a URL that never actually exists. Lowercase
// every generated file name and internal link so both sides agree.
const fs = require("fs");
const path = require("path");

const refDir = process.argv[2];
if (!refDir) {
    console.error("Usage: node lowercase-typedoc-refs.js <reference-dir>");
    process.exit(1);
}

function walk(dir) {
    let files = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) files = files.concat(walk(full));
        else if (entry.name.endsWith(".md")) files.push(full);
    }
    return files;
}

const files = walk(refDir);

for (const file of files) {
    const text = fs.readFileSync(file, "utf8");
    const fixed = text.replace(/\]\(([^)]+\.md(?:#[^)]*)?)\)/g, (match, url) => {
        const lower = url.toLowerCase();
        return `](${lower.replace(/\.md(?=$|#)/, "")})`;
    });
    if (fixed !== text) fs.writeFileSync(file, fixed);
}

// Rename deepest paths first so renaming a directory doesn't invalidate
// already-computed paths of files nested inside it.
files.sort((a, b) => b.length - a.length);
for (const file of files) {
    const lower = file.toLowerCase();
    if (lower !== file) {
        // Case-only renames are a no-op on case-insensitive filesystems
        // (Windows/macOS) unless done via an intermediate name.
        const tmp = file + ".__tmp__";
        fs.renameSync(file, tmp);
        fs.renameSync(tmp, lower);
    }
}

console.log(`Lowercased ${files.length} TypeDoc reference file(s) and their internal links.`);

