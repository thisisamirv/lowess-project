[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / Lowess

# Class: Lowess

Defined in: [index.d.ts:4](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L4)

Batch LOWESS smoothing.

## Constructors

### Constructor

> **new Lowess**(`options?`): `Lowess`

Defined in: [index.d.ts:6](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L6)

Create a new batch LOWESS smoother.

#### Parameters

##### options?

[`SmoothOptions`](../interfaces/SmoothOptions.md) \| `null`

#### Returns

`Lowess`

## Methods

### fit()

> **fit**(`x`, `y`, `customWeights?`): [`LowessResult`](LowessResult.md)

Defined in: [index.d.ts:8](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L8)

Fit the model.

#### Parameters

##### x

`Float64Array`

##### y

`Float64Array`

##### customWeights?

`Float64Array`\<`ArrayBufferLike`\> \| `null`

#### Returns

[`LowessResult`](LowessResult.md)

***

### fit\_async()

> **fit\_async**(`x`, `y`, `customWeights?`): `Promise`\<`unknown`\>

Defined in: [index.d.ts:10](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L10)

Fit the model asynchronously.

#### Parameters

##### x

`Float64Array`

##### y

`Float64Array`

##### customWeights?

`Float64Array`\<`ArrayBufferLike`\> \| `null`

#### Returns

`Promise`\<`unknown`\>
