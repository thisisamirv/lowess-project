'use strict';

// Embeds the binding's top-level README.md as the body of the Starlight
// splash homepage (src/content/docs/index.md), below its existing frontmatter.

const fs = require('fs');
const path = require('path');

// Accept binding dir as first argument; fall back to cwd
const BINDING_DIR = process.argv[2] ? path.resolve(process.argv[2]) : process.cwd();
const INDEX_PATH = path.join(BINDING_DIR, 'src', 'content', 'docs', 'index.md');
const README_PATH = path.join(BINDING_DIR, 'README.md');

const indexContent = fs.readFileSync(INDEX_PATH, 'utf-8').replace(/\r\n/g, '\n');
const frontmatterMatch = indexContent.match(/^---\n[\s\S]*?\n---\n/);
if (!frontmatterMatch) {
    throw new Error(`No frontmatter block found in ${INDEX_PATH}`);
}
const frontmatter = frontmatterMatch[0];
const readme = fs.readFileSync(README_PATH, 'utf-8')
    .replace(/\r\n/g, '\n')
    // Starlight's hero already shows the title; drop the README's H1 so it isn't duplicated.
    .replace(/^(<!--[^\n]*-->\n)?# .+\n\n?/, '$1\n');

fs.writeFileSync(INDEX_PATH, `${frontmatter}\n${readme}`, 'utf-8');
console.log(`Embedded README.md into ${path.relative(BINDING_DIR, INDEX_PATH)}`);
