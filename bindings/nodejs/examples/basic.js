const fastlowess = require('..');

// 1. Prepare data
const x = new Float64Array(100).map((_, i) => i / 10);
const y = new Float64Array(100).map((_, i) => Math.sin(x[i]) + (Math.random() - 0.5) * 0.2);

// 2. Simple Batch Smoothing
console.log("--- Batch Smoothing ---");
const result = fastlowess.smooth(x, y, {
    fraction: 0.3,
    returnDiagnostics: true
});
console.log(`RMSE: ${result.diagnostics.rmse.toFixed(4)}`);
console.log(`First 5 smoothed values: ${result.y.slice(0, 5)}`);

// 3. Streaming Smoothing
console.log("\n--- Streaming Smoothing ---");
const streamer = new fastlowess.StreamingLowess({ fraction: 0.3 }, { chunkSize: 50, overlap: 10 });
const r1 = streamer.processChunk(x.slice(0, 50), y.slice(0, 50));
const r2 = streamer.processChunk(x.slice(50, 100), y.slice(50, 100));
const final = streamer.finalize();
console.log(`Total smoothed points: ${r1.y.length + r2.y.length + final.y.length}`);

// 4. Online Smoothing
console.log("\n--- Online Smoothing ---");
const online = new fastlowess.OnlineLowess({ fraction: 0.3 }, { windowCapacity: 20 });
for (let i = 0; i < 10; i++) {
    const val = online.update(x[i], y[i]);
    if (val !== null) {
        console.log(`Point ${i}: ${val.toFixed(4)}`);
    }
}
