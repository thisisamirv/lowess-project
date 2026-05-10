#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

function usage() {
    console.error("Usage: node dev/check_js_licenses.mjs <package-dir> [--fail-on-gpl]");
    process.exit(2);
}

const args = process.argv.slice(2);
if (args.length < 1) {
    usage();
}

const packageDir = path.resolve(args[0]);
const failOnGpl = args.includes("--fail-on-gpl");
const nodeModulesDir = path.join(packageDir, "node_modules");

if (!fs.existsSync(nodeModulesDir)) {
    console.error(`node_modules not found at ${nodeModulesDir}`);
    process.exit(1);
}

const seenPackages = new Set();
const licenseCounts = new Map();
const gplPackages = [];

function normalizeLicense(licenseValue) {
    if (!licenseValue) {
        return "UNKNOWN";
    }

    if (typeof licenseValue === "string") {
        return licenseValue;
    }

    if (Array.isArray(licenseValue)) {
        return licenseValue.map(normalizeLicense).join(" OR ");
    }

    if (typeof licenseValue === "object") {
        if (typeof licenseValue.type === "string") {
            return licenseValue.type;
        }
        if (typeof licenseValue.name === "string") {
            return licenseValue.name;
        }
    }

    return "UNKNOWN";
}

function recordPackage(packageJsonPath) {
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
    const packageName = packageJson.name;
    const packageVersion = packageJson.version;

    if (!packageName || !packageVersion) {
        return;
    }

    const packageKey = `${packageName}@${packageVersion}`;
    if (seenPackages.has(packageKey)) {
        return;
    }
    seenPackages.add(packageKey);

    const license = normalizeLicense(packageJson.license);
    licenseCounts.set(license, (licenseCounts.get(license) ?? 0) + 1);

    if (/\b(?:A?GPL|LGPL|GPL)\b/i.test(license)) {
        gplPackages.push(`${packageKey}: ${license}`);
    }
}

function walkNodeModules(dirPath) {
    for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
        if (!entry.isDirectory()) {
            continue;
        }
        if (entry.name === ".bin") {
            continue;
        }

        const entryPath = path.join(dirPath, entry.name);

        if (entry.name.startsWith("@")) {
            walkNodeModules(entryPath);
            continue;
        }

        const packageJsonPath = path.join(entryPath, "package.json");
        if (fs.existsSync(packageJsonPath)) {
            recordPackage(packageJsonPath);
        }

        const nestedNodeModules = path.join(entryPath, "node_modules");
        if (fs.existsSync(nestedNodeModules)) {
            walkNodeModules(nestedNodeModules);
        }
    }
}

walkNodeModules(nodeModulesDir);

const entries = [...licenseCounts.entries()].sort((left, right) => {
    if (right[1] !== left[1]) {
        return right[1] - left[1];
    }
    return left[0].localeCompare(right[0]);
});

entries.forEach(([license, count], index) => {
    const prefix = index === entries.length - 1 ? "└─" : "├─";
    console.log(`${prefix} ${license}: ${count}`);
});

if (failOnGpl && gplPackages.length > 0) {
    console.error("Detected disallowed GPL-family licenses:");
    gplPackages.sort().forEach(pkg => console.error(`- ${pkg}`));
    process.exit(1);
}