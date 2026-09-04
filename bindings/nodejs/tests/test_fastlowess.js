const test = require('node:test');
const assert = require('node:assert');

const fastlowess = require('..');

test('batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        fraction: 0.3,
        return_diagnostics: true
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
        return_residuals: true,
        return_robustness_weights: true,
        return_sorted: true
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
        return_residuals: true,
        return_robustness_weights: true
    });
    const unsortedResult = unsortedModel.fit(x, y);

    const sortedPairs = Array.from(result.x).map((xv, i) => [xv, result.y[i]]).sort();
    const unsortedPairs = Array.from(unsortedResult.x).map((xv, i) => [xv, unsortedResult.y[i]]).sort();
    assert.deepStrictEqual(sortedPairs, unsortedPairs);

    assert.strictEqual(result.residuals.length, x.length);
    assert.strictEqual(result.robustness_weights.length, x.length);
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
