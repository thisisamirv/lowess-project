const test = require('node:test');
const assert = require('node:assert');

// Import WASM bindings using require (works in Node with generated pkg)
const fastlowess = require('../../bindings/wasm/pkg/fastlowess_wasm.js');

test('WASM batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastlowess.Lowess({
        fraction: 0.3,
        return_diagnostics: true
    }).fit(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    // Check diagnostics using getters
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('WASM streaming smoothing', () => {
    const streamer = new fastlowess.StreamingLowess({
        fraction: 0.3
    }, {
        chunk_size: 10,
        overlap: 2
    });

    const x = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2));

    const result = streamer.process_chunk(x, y);
    // WASM processChunk returns a struct, safe to check .y existence/length if populated
    if (result) {
        assert.ok(result.y.length >= 0);
    }

    const finalResult = streamer.finalize();
    if (finalResult) {
        assert.ok(finalResult.y.length > 0);
    }
});

test('WASM online smoothing', () => {
    const online = new fastlowess.OnlineLowess({
        fraction: 0.5
    }, {
        window_capacity: 10,
        min_points: 2
    });

    let lastSmoothed;
    for (let i = 0; i < 10; i++) {
        const res = online.add_point(i, i * 2);
        if (res !== undefined) {
            lastSmoothed = res.smoothed;
        }
    }

    // lastVal should not be undefined/null after enough points
    assert.ok(lastSmoothed !== undefined && lastSmoothed !== null);
    assert.ok(Math.abs(lastSmoothed - 18) < 1.0);
});

test('WASM options parsing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastlowess.Lowess({
        weight_function: 'tricube',
        robustness_method: 'bisquare',
        boundary_policy: 'extend',
        scaling_method: 'mad'
    }).fit(x, y);

    assert.strictEqual(result.y.length, 5);
});

test('WASM custom_weights: uniform weights match no weights', () => {
    const n = 20;
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i * 0.5));
    const y = new Float64Array(x.map(v => Math.sin(v)));
    const weights = Array.from({ length: n }, () => 1.0);

    const result_no_w = new fastlowess.Lowess({ fraction: 0.4, iterations: 2 }).fit(x, y);
    const result_unit_w = new fastlowess.Lowess({ fraction: 0.4, iterations: 2 }).fit(x, y, weights);

    for (let i = 0; i < n; i++) {
        assert.ok(
            Math.abs(result_no_w.y[i] - result_unit_w.y[i]) < 1e-10,
            `y[${i}] diverges: ${result_no_w.y[i]} vs ${result_unit_w.y[i]}`
        );
    }
});

test('WASM custom_weights: zero weight reduces outlier influence', () => {
    const n = 10;
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i));
    const y = new Float64Array(x.map(v => v * 2.0));
    y[5] = 100.0;  // outlier

    const weights = Array.from({ length: n }, () => 1.0);
    weights[5] = 0.0;

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

test('WASM custom_weights: wrong length throws error', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    assert.throws(() => {
        new fastlowess.Lowess({ fraction: 0.5 }).fit(x, y, [1, 1, 1]);
    });
});
