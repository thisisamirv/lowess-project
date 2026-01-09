use fastlowess_wasm::*;
use js_sys::Float64Array;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_smooth() {
    let x = Float64Array::from(&[1.0, 2.0, 3.0, 4.0, 5.0][..]);
    let y = Float64Array::from(&[2.0, 4.0, 6.0, 8.0, 10.0][..]);

    let options = JsValue::NULL;
    let result = smooth(&x, &y, &options).expect("Smooth failed");

    assert_eq!(result.x().length(), 5);
    assert_eq!(result.y().length(), 5);
}

#[wasm_bindgen_test]
fn test_streaming() {
    let mut streamer = StreamingLowessWasm::new(&JsValue::NULL, &JsValue::NULL)
        .expect("Failed to create streamer");

    let x = Float64Array::from(&[1.0, 2.0][..]);
    let y = Float64Array::from(&[2.0, 4.0][..]);

    let result = streamer
        .process_chunk(&x, &y)
        .expect("Process chunk failed");
    // For small data it might return empty result due to buffering
    assert!(result.y().length() < 10);

    let final_result = streamer.finalize().expect("Finalize failed");
    assert_eq!(final_result.y().length(), 2);
}

#[wasm_bindgen_test]
fn test_online() {
    let mut online =
        OnlineLowessWasm::new(&JsValue::NULL, &JsValue::NULL).expect("Failed to create online");

    let mut last_val = None;
    for i in 0..10 {
        last_val = online
            .update(i as f64, (i * 2) as f64)
            .expect("Update failed");
    }

    assert!(last_val.is_some());
}
