const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const fastlowess = require('..');
const nativeLoaderSource = fs.readFileSync(path.join(__dirname, '..', 'index.js'), 'utf8');
const gpuInstallerTesting = require('../gpu-installer')._testing;

test('version metadata is available without a native addon', () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'fastlowess-version-'));
    try {
        const packageDirectory = path.join(directory, 'node_modules', 'fastlowess');
        fs.mkdirSync(packageDirectory, { recursive: true });
        for (const filename of ['package.json', 'version.js']) {
            fs.copyFileSync(
                path.join(__dirname, '..', filename),
                path.join(packageDirectory, filename)
            );
        }
        const child = spawnSync(process.execPath, [
            '-e',
            "process.stdout.write(require('fastlowess/version').version)",
        ], { cwd: directory, encoding: 'utf8' });
        assert.strictEqual(child.status, 0, child.stderr);
        assert.strictEqual(child.stdout, require('../package.json').version);
        assert.strictEqual(require('../version').version, child.stdout);
    } finally {
        fs.rmSync(directory, { recursive: true, force: true });
    }
});

test('native loader version checks follow package metadata', () => {
    assert.match(
        nativeLoaderSource,
        /bindingPackageVersion !== require\('\.\/package\.json'\)\.version/
    );
    assert.doesNotMatch(nativeLoaderSource, /bindingPackageVersion !== '\d+\.\d+\.\d+'/);
    assert.doesNotMatch(nativeLoaderSource, /expected \d+\.\d+\.\d+ but got/);
});

test('GPU target detection handles musl reports and rejects unsupported ARM musl', () => {
    assert.strictEqual(
        gpuInstallerTesting.isMuslFromReport({
            header: {},
            sharedObjects: ['/lib/ld-musl-x86_64.so.1'],
        }),
        true
    );
    assert.strictEqual(
        gpuInstallerTesting.isMuslFromReport({
            header: { glibcVersionRuntime: '2.39' },
            sharedObjects: [],
        }),
        false
    );
    assert.strictEqual(
        gpuInstallerTesting.currentPlatformSuffix('linux', 'x64', true),
        'linux-x64-musl'
    );
    assert.strictEqual(gpuInstallerTesting.currentPlatformSuffix('linux', 'arm', true), null);
});

test('unknown outputs are rejected for each API mode', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);
    const invalidOutput = /unknown output/i;

    assert.throws(() => new fastlowess.Lowess({ outputs: ['typo'] }).fit(x, y), invalidOutput);
    assert.throws(() => new fastlowess.StreamingLowess({ outputs: ['sorted'] }), invalidOutput);
    assert.throws(() => new fastlowess.OnlineLowess({ outputs: ['diagnostics'] }), invalidOutput);

    const result = new fastlowess.Lowess({ retain_model: true }).fit(x, y);
    assert.throws(() => result.predict(x, { outputs: ['weights'] }), invalidOutput);
});

test('GPU installer rejects a CPU-only local addon', async () => {
    await assert.rejects(
        fastlowess.installGpu({ yes: true, localPath: require.resolve('..') }),
        /does not report GPU support/
    );
});

test('GPU installer rejects an active native-library override', async () => {
    const previousOverride = process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fastlowess-addon-'));
    const overridePath = path.join(tempDir, 'cpu-addon.js');
    fs.writeFileSync(overridePath, 'module.exports = { gpu_enabled: () => false };\n');
    process.env.NAPI_RS_NATIVE_LIBRARY_PATH = overridePath;
    try {
        await assert.rejects(
            fastlowess.installGpu({ yes: true, localPath: require.resolve('..') }),
            /NAPI_RS_NATIVE_LIBRARY_PATH overrides/
        );
    } finally {
        if (previousOverride === undefined) {
            delete process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
        } else {
            process.env.NAPI_RS_NATIVE_LIBRARY_PATH = previousOverride;
        }
        fs.rmSync(tempDir, { recursive: true, force: true });
    }
});

test('batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        fraction: 0.3,
        outputs: ["diagnostics"]
    });

    const result = model.fit(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('streaming smoothing', () => {
    const streamer = new fastlowess.StreamingLowess({
        fraction: 0.3
    }, {
        chunk_size: 10,
        overlap: 2
    });

    const x = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2));

    const result = streamer.process_chunk(x, y);
    assert.ok(result.y.length >= 0);

    const finalResult = streamer.finalize();
    assert.ok(finalResult.y.length > 0);
});

test('online smoothing', () => {
    const online = new fastlowess.OnlineLowess({
        fraction: 0.5
    }, {
        window_capacity: 10,
        min_points: 2
    });

    let lastVal = null;
    for (let i = 0; i < 10; i++) {
        const res = online.add_point(i, i * 2);

        if (res !== null) {
            lastVal = res.y;
        }
    }

    assert.ok(lastVal !== null);
    assert.ok(Math.abs(lastVal - 18) < 1.0);
});

test('options parsing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        weight_function: 'tricube',
        robustness_method: 'bisquare',
        boundary_policy: 'extend',
        scaling_method: 'mad'
    });

    const result = model.fit(x, y);

    assert.strictEqual(result.y.length, 5);
});

test('return_sorted defaults to original input order', () => {
    const x = new Float64Array([3, 1, 5, 2, 4]);
    const y = new Float64Array([6, 2, 10, 4, 8]);

    const model = new fastlowess.Lowess({ fraction: 0.7 });
    const result = model.fit(x, y);

    assert.deepStrictEqual(Array.from(result.x), Array.from(x));
});

test('return_sorted = true returns results sorted ascending by x', () => {
    const x = new Float64Array([3, 1, 5, 2, 4]);
    const y = new Float64Array([6, 2, 10, 4, 8]);

    const model = new fastlowess.Lowess({
        fraction: 0.7,
        outputs: ["residuals", "weights", "sorted"]
    });
    const result = model.fit(x, y);

    // x must be strictly ascending, and differ from the unsorted input order.
    for (let i = 1; i < result.x.length; i++) {
        assert.ok(result.x[i - 1] <= result.x[i]);
    }
    assert.notDeepStrictEqual(Array.from(result.x), Array.from(x));

    // Same (x, y) pairs as the unsorted-order fit, just reordered.
    const unsortedModel = new fastlowess.Lowess({
        fraction: 0.7,
        outputs: ["residuals", "weights"]
    });
    const unsortedResult = unsortedModel.fit(x, y);

    const sortedPairs = Array.from(result.x).map((xv, i) => [xv, result.y[i]]).sort();
    const unsortedPairs = Array.from(unsortedResult.x).map((xv, i) => [xv, unsortedResult.y[i]]).sort();
    assert.deepStrictEqual(sortedPairs, unsortedPairs);

    assert.strictEqual(result.residuals.length, x.length);
    assert.strictEqual(result.robustness_weights.length, x.length);
});

test('SmoothOptions: return_derivative returns per-point local fit slope', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        fraction: 0.7,
        outputs: ["derivative"],
    });
    const result = model.fit(x, y);

    assert.ok(result.derivative !== null);
    assert.strictEqual(result.derivative.length, x.length);
});

test('SmoothOptions: derivative is null when return_derivative not requested', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({ fraction: 0.7 });
    const result = model.fit(x, y);

    assert.strictEqual(result.derivative, null);
});

test('async batch smoothing', async () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        fraction: 0.3
    });

    if (typeof model.fit_async !== 'function') {
        console.error('Available properties on model:', Object.getOwnPropertyNames(Object.getPrototypeOf(model)));
        throw new Error('model.fit_async is not a function');
    }
    const result = await model.fit_async(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    assert.ok(result.y[0] > 0);
});

test('custom_weights: uniform weights match no weights', () => {
    const n = 20;
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i * 0.5));
    const y = new Float64Array(x.map(v => Math.sin(v)));
    const weights = new Float64Array(n).fill(1.0);

    const result_no_w = new fastlowess.Lowess({ fraction: 0.4, iterations: 2 }).fit(x, y);
    const result_unit_w = new fastlowess.Lowess({ fraction: 0.4, iterations: 2 }).fit(x, y, weights);

    for (let i = 0; i < n; i++) {
        assert.ok(
            Math.abs(result_no_w.y[i] - result_unit_w.y[i]) < 1e-10,
            `y[${i}] diverges: ${result_no_w.y[i]} vs ${result_unit_w.y[i]}`
        );
    }
});

test('custom_weights: zero weight reduces outlier influence', () => {
    const n = 10;
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i));
    const y = new Float64Array(x.map(v => v * 2.0));
    y[5] = 100.0;  // outlier

    const weights = new Float64Array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]);

    const result_no_w = new fastlowess.Lowess({ fraction: 0.5, iterations: 0 }).fit(x, y);
    const result_zero_w = new fastlowess.Lowess({ fraction: 0.5, iterations: 0 }).fit(x, y, weights);

    const true_val = 5.0 * 2.0;
    const err_no_w = Math.abs(result_no_w.y[5] - true_val);
    const err_zero_w = Math.abs(result_zero_w.y[5] - true_val);

    assert.ok(
        err_zero_w < err_no_w,
        `zero weight should reduce error (no_w=${err_no_w.toFixed(2)}, zero_w=${err_zero_w.toFixed(2)})`
    );
});

test('custom_weights: wrong length throws error', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    assert.throws(() => {
        new fastlowess.Lowess({ fraction: 0.5 }).fit(x, y, new Float64Array([1, 1, 1]));
    });
});

test('missing: default ("error") throws on NaN', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, NaN, 6, 8, 10]);

    assert.throws(() => {
        new fastlowess.Lowess({ fraction: 0.5 }).fit(x, y);
    });
});

test('missing: "drop" removes non-finite rows', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, NaN, 6, 8, 10]);

    const model = new fastlowess.Lowess({ fraction: 0.5, missing: 'drop' });
    const result = model.fit(x, y);

    assert.strictEqual(result.y.length, x.length - 1);
});

test('missing: invalid policy throws', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    assert.throws(() => {
        new fastlowess.Lowess({ fraction: 0.5, missing: 'invalid' }).fit(x, y);
    });
});

test('streaming missing: "drop" removes non-finite rows', () => {
    const streamer = new fastlowess.StreamingLowess({
        fraction: 0.1,
        missing: 'drop'
    }, {
        chunk_size: 50
    });

    const x = new Float64Array(Array.from({ length: 50 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 50 }, (_, i) => i));
    y[5] = NaN;

    const result = streamer.process_chunk(x, y);
    const finalResult = streamer.finalize();

    assert.strictEqual(result.y.length + finalResult.y.length, x.length - 1);
});

test('online missing: "drop" ignores non-finite point', () => {
    const online = new fastlowess.OnlineLowess({
        fraction: 0.5,
        missing: 'drop'
    }, {
        window_capacity: 10
    });

    const res = online.add_point(1, NaN);
    assert.strictEqual(res, null);
});

test('streaming: return_se and grouped intervals', () => {
    const streamer = new fastlowess.StreamingLowess({
        fraction: 0.2,
        outputs: ["se"],
        intervals: { confidence: 0.95, prediction: 0.95 }
    }, {
        chunk_size: 50
    });

    const n = 200;
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: n }, (_, i) => Math.sin(i / 10)));

    const result = streamer.process_chunk(x, y);
    const finalResult = streamer.finalize();

    assert.ok(result.standard_errors !== null);
    assert.ok(result.confidence_lower !== null);
    assert.ok(result.confidence_upper !== null);
    assert.ok(result.prediction_lower !== null);
    assert.ok(result.prediction_upper !== null);
    assert.ok(finalResult.standard_errors !== null);
});

test('online: return_se and intervals require update_mode "full"', () => {
    assert.throws(() => {
        new fastlowess.OnlineLowess({
            fraction: 0.5,
            outputs: ["se"]
        }, {
            window_capacity: 10,
            min_points: 3
        });
    });
});

test('online: return_se and grouped intervals', () => {
    const online = new fastlowess.OnlineLowess({
        fraction: 0.5,
        outputs: ["se"],
        intervals: { confidence: 0.95, prediction: 0.95 }
    }, {
        window_capacity: 10,
        min_points: 3,
        update_mode: 'full'
    });

    let last = null;
    for (let i = 0; i < 10; i++) {
        const res = online.add_point(i, i * 2);
        if (res !== null) {
            last = res;
        }
    }

    assert.ok(last !== null);
    assert.ok(last.standard_error !== null);
    assert.ok(last.confidence_lower !== null);
    assert.ok(last.confidence_upper !== null);
    assert.ok(last.prediction_lower !== null);
    assert.ok(last.prediction_upper !== null);
});

function wavy(n, step = 1) {
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i * step));
    const y = x.map((v) => Math.sin(v * 0.1) + 0.1 * Math.cos(0.7 * v));
    return { x, y };
}

const bootstrapIntervals = { confidence: 0.95, prediction: 0.95, bootstrap: 20 };

test('batch: shared seed reproduces CV scores and bootstrap intervals', () => {
    const { x, y } = wavy(40);
    const runs = [0, 1].map(() => new fastlowess.Lowess({
        intervals: bootstrapIntervals,
        cv: { method: 'kfold', k: 4, fractions: [0.3, 0.5, 0.7] },
        seed: 0
    }).fit(x, y));
    assert.deepStrictEqual(runs[0].cv_scores, runs[1].cv_scores);
    assert.deepStrictEqual(runs[0].confidence_lower, runs[1].confidence_lower);
    assert.strictEqual(runs[0].standard_errors.length, x.length);
    assert.strictEqual(runs[0].prediction_upper.length, x.length);
});

test('batch: negative seed and single bootstrap replicate are rejected', () => {
    const { x, y } = wavy(20);
    assert.throws(() => new fastlowess.Lowess({ seed: -1 }).fit(x, y));
    assert.throws(() => new fastlowess.Lowess({
        intervals: { confidence: 0.95, bootstrap: 1 }
    }).fit(x, y));
});

test('streaming: seeded bootstrap intervals', () => {
    const { x, y } = wavy(30, 0.1);
    const lower = [0, 1].map(() => new fastlowess.StreamingLowess(
        { intervals: bootstrapIntervals, seed: 42 },
        { chunk_size: x.length }
    ).process_chunk(x, y).confidence_lower);
    assert.ok(lower[0] !== null);
    assert.deepStrictEqual(lower[0], lower[1]);
});

test('online: seeded bootstrap intervals require update_mode "full"', () => {
    const { x, y } = wavy(12, 0.1);
    const bounds = [0, 1].map(() => {
        const online = new fastlowess.OnlineLowess(
            { fraction: 0.5, intervals: bootstrapIntervals, seed: 0 },
            { window_capacity: 20, min_points: 5, update_mode: 'full' }
        );
        let last = null;
        for (let i = 0; i < x.length; i++) {
            last = online.add_point(x[i], y[i]) ?? last;
        }
        return [last.confidence_lower, last.prediction_upper];
    });
    assert.ok(bounds[0].every((v) => v !== null));
    assert.deepStrictEqual(bounds[0], bounds[1]);
    assert.throws(() => new fastlowess.OnlineLowess({ intervals: bootstrapIntervals }));
});

test('predict: grouped outputs, intervals, and seeded bootstrap', () => {
    const { x, y } = wavy(30);
    const result = new fastlowess.Lowess({ retain_model: true }).fit(x, y);
    const newX = new Float64Array([2.5, 10.5, 20.5]);

    const analytic = result.predict(newX, {
        outputs: ['se', 'derivative'],
        intervals: { confidence: 0.95, prediction: 0.95 }
    });
    assert.strictEqual(analytic.standard_errors.length, newX.length);
    assert.strictEqual(analytic.derivative.length, newX.length);
    assert.strictEqual(analytic.prediction_lower.length, newX.length);

    const opts = { intervals: { confidence: 0.95, bootstrap: 20 }, seed: 7 };
    assert.deepStrictEqual(
        result.predict(newX, opts).confidence_lower,
        result.predict(newX, opts).confidence_lower
    );
});
