const fastlowess = require('../../bindings/nodejs');

/**
 * fastlowess Batch Smoothing Example
 * 
 * This example demonstrates batch LOWESS smoothing features:
 * - Basic smoothing with different parameters
 * - Robustness iterations for outlier handling
 * - Confidence and prediction intervals
 * - Diagnostics and cross-validation
 *
 * The batch adapter (smooth function) is the primary interface for
 * processing complete datasets that fit in memory.
 */

function generateSampleData(nPoints = 1000) {
    // Generate complex sample data with a trend, seasonality, and outliers.
    const x = new Float64Array(nPoints);
    const y = new Float64Array(nPoints);
    const yTrue = new Float64Array(nPoints);

    for (let i = 0; i < nPoints; i++) {
        x[i] = (i / (nPoints - 1)) * 50; // range 0 to 50
        // Trend + Seasonality
        yTrue[i] = 0.5 * x[i] + 5 * Math.sin(x[i] * 0.5);
        // Gaussian noise (approx)
        let u1 = Math.random();
        let u2 = Math.random();
        let z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        y[i] = yTrue[i] + z * 1.5;
    }

    // Add significant outliers (10% of data)
    const nOutliers = Math.round(nPoints * 0.1);
    for (let k = 0; k < nOutliers; k++) {
        const idx = Math.floor(Math.random() * nPoints);
        const sign = Math.random() < 0.5 ? -1 : 1;
        y[idx] += (10 + Math.random() * 10) * sign;
    }

    return { x, y, yTrue };
}

function main() {
    console.log("=== fastlowess Batch Smoothing Example ===");

    // 1. Generate Data
    const { x, y } = generateSampleData(1000);
    console.log(`Generated ${x.length} data points with outliers.`);

    // 2. Basic Smoothing (Default parameters)
    console.log("Running basic smoothing...");
    // Use a smaller fraction (0.05) to capture the sine wave seasonality
    const resBasic = fastlowess.smooth(x, y, { iterations: 0, fraction: 0.05 });

    // 3. Robust Smoothing (IRLS)
    console.log("Running robust smoothing (3 iterations)...");
    const resRobust = fastlowess.smooth(x, y, {
        fraction: 0.05,
        iterations: 3,
        robustnessMethod: "bisquare",
        returnRobustnessWeights: true
    });

    // 4. Uncertainty Quantification
    console.log("Computing confidence and prediction intervals...");
    const resIntervals = fastlowess.smooth(x, y, {
        fraction: 0.05,
        confidenceIntervals: 0.95,
        predictionIntervals: 0.95,
        returnDiagnostics: true
    });

    // 5. Cross-Validation for optimal fraction
    console.log("Running cross-validation to find optimal fraction...");
    const cvFractions = [0.05, 0.1, 0.2, 0.4];
    
    // We pass the fractions and CV method to the smooth function.
    // The main result returned will be the fit using the BEST fraction.
    // We can also retrieve the score for each fraction.
    const resCV = fastlowess.smooth(x, y, { 
        cvFractions, 
        cvMethod: "kfold", 
        cvK: 5,
        // We can request robustness weights or diagnostics for the final best model too
        returnDiagnostics: true 
    });

    console.log(`Optimal fraction selected: ${resCV.fractionUsed}`);
    
    // Check scores if available
    const scores = resCV.cvScores;
    // Note: cvScores array corresponds to the input fractions order.
    if (scores) {
        console.log("CV Scores (RMSE):");
        for(let i=0; i<cvFractions.length; i++) {
            console.log(` - Fraction ${cvFractions[i]}: ${scores[i].toFixed(4)}`);
        }
    }
    // Diagnostics Printout
    if (resIntervals.diagnostics) {
        const diag = resIntervals.diagnostics;
        console.log("\nFit Statistics (Intervals Model):");
        console.log(` - R²:   ${diag.rSquared.toFixed(4)}`);
        console.log(` - RMSE: ${diag.rmse.toFixed(4)}`);
        console.log(` - MAE:  ${diag.mae.toFixed(4)}`);
    }

    // 6. Boundary Policy Comparison
    console.log("\nDemonstrating boundary policy effects on linear data...");
    const xl = new Float64Array(50);
    const yl = new Float64Array(50);
    for(let i=0; i<50; i++) {
        xl[i] = (i / 49) * 10;
        yl[i] = 2 * xl[i] + 1;
    }

    const rExt = fastlowess.smooth(xl, yl, { fraction: 0.6, boundaryPolicy: "extend" });
    const rRef = fastlowess.smooth(xl, yl, { fraction: 0.6, boundaryPolicy: "reflect" });
    const rZr  = fastlowess.smooth(xl, yl, { fraction: 0.6, boundaryPolicy: "zero" });

    console.log("Boundary policy comparison:");
    console.log(` - Extend (Default): first=${rExt.y[0].toFixed(2)}, last=${rExt.y[49].toFixed(2)}`);
    console.log(` - Reflect:          first=${rRef.y[0].toFixed(2)}, last=${rRef.y[49].toFixed(2)}`);
    console.log(` - Zero:             first=${rZr.y[0].toFixed(2)}, last=${rZr.y[49].toFixed(2)}`);

    console.log("\n=== Batch Smoothing Example Complete ===");
}

main();
