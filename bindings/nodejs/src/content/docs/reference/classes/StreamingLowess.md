[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / StreamingLowess

# Class: StreamingLowess

Defined in: [index.d.ts:52](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L52)

Streaming LOWESS smoother for large datasets.

## Constructors

### Constructor

> **new StreamingLowess**(`options?`, `streamingOpts?`): `StreamingLowess`

Defined in: [index.d.ts:54](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L54)

Create a new streaming LOWESS smoother.

#### Parameters

##### options?

[`SmoothOptions`](../interfaces/SmoothOptions.md) \| `null`

##### streamingOpts?

[`StreamingOptions`](../interfaces/StreamingOptions.md) \| `null`

#### Returns

`StreamingLowess`

## Methods

### finalize()

> **finalize**(): [`LowessResult`](LowessResult.md)

Defined in: [index.d.ts:58](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L58)

Finalize the stream and return remaining data.

#### Returns

[`LowessResult`](LowessResult.md)

***

### process\_chunk()

> **process\_chunk**(`x`, `y`): [`LowessResult`](LowessResult.md)

Defined in: [index.d.ts:56](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L56)

Process a chunk of data.

#### Parameters

##### x

`Float64Array`

##### y

`Float64Array`

#### Returns

[`LowessResult`](LowessResult.md)
