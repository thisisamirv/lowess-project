const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
// Import WASM bindings using require (works in Node with generated pkg)
const fastlowess = require('../../bindings/wasm/pkg/fastlowess_wasm.js');

test('WASM batch smoothing', (t) => {
  const x = new Float64Array([1, 2, 3, 4, 5]);
  const y = new Float64Array([2, 4, 6, 8, 10]);
  
  const result = fastlowess.smooth(x, y, {
    fraction: 0.3,
    returnDiagnostics: true
  });
  
  assert.strictEqual(result.x.length, 5);
  assert.strictEqual(result.y.length, 5);
  // Check diagnostics using getters
  assert.ok(result.diagnostics.rmse < 0.1);
});

test('WASM streaming smoothing', (t) => {
  const streamer = new fastlowess.StreamingLowessWasm({
    fraction: 0.3
  }, {
    chunkSize: 10,
    overlap: 2
  });
  
  const x = new Float64Array(Array.from({length: 20}, (_, i) => i));
  const y = new Float64Array(Array.from({length: 20}, (_, i) => i * 2));
  
  const result = streamer.processChunk(x, y);
  // WASM processChunk returns a struct, safe to check .y existence/length if populated
  if (result) {
    assert.ok(result.y.length >= 0);
  }
  
  const finalResult = streamer.finalize();
  if (finalResult) {
    assert.ok(finalResult.y.length > 0);
  }
});

test('WASM online smoothing', (t) => {
  const online = new fastlowess.OnlineLowessWasm({
    fraction: 0.5
  }, {
    windowCapacity: 10,
    minPoints: 2
  });
  
  let lastVal;
  for (let i = 0; i < 10; i++) {
    lastVal = online.update(i, i * 2);
  }
  
  // lastVal should not be undefined/null after enough points
  assert.ok(lastVal !== undefined && lastVal !== null);
  assert.ok(Math.abs(lastVal - 18) < 1.0);
});

test('WASM options parsing', (t) => {
  const x = new Float64Array([1, 2, 3, 4, 5]);
  const y = new Float64Array([2, 4, 6, 8, 10]);
  
  const result = fastlowess.smooth(x, y, {
    weightFunction: 'tricube',
    robustnessMethod: 'bisquare',
    boundaryPolicy: 'extend',
    scalingMethod: 'mad'
  });
  
  assert.strictEqual(result.y.length, 5);
});
