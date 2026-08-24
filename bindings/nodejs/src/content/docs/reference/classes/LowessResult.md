[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / LowessResult

# Class: LowessResult

Defined in: [index.d.ts:14](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L14)

Result of a LOWESS fit.

## Constructors

### Constructor

> **new LowessResult**(): `LowessResult`

#### Returns

`LowessResult`

## Accessors

### confidence\_lower

#### Get Signature

> **get** **confidence\_lower**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:24](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L24)

Get lower confidence bounds (if requested).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### confidence\_upper

#### Get Signature

> **get** **confidence\_upper**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:26](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L26)

Get upper confidence bounds (if requested).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### cv\_scores

#### Get Signature

> **get** **cv\_scores**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:36](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L36)

Get cross-validation scores (if CV was performed).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### diagnostics

#### Get Signature

> **get** **diagnostics**(): [`Diagnostics`](../interfaces/Diagnostics.md) \| `null`

Defined in: [index.d.ts:34](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L34)

Get diagnostics (if requested).

##### Returns

[`Diagnostics`](../interfaces/Diagnostics.md) \| `null`

***

### fraction\_used

#### Get Signature

> **get** **fraction\_used**(): `number`

Defined in: [index.d.ts:38](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L38)

Get the fraction used for smoothing.

##### Returns

`number`

***

### iterations\_used

#### Get Signature

> **get** **iterations\_used**(): `number` \| `null`

Defined in: [index.d.ts:40](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L40)

Get the number of iterations performed.

##### Returns

`number` \| `null`

***

### prediction\_lower

#### Get Signature

> **get** **prediction\_lower**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:28](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L28)

Get lower prediction bounds (if requested).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### prediction\_upper

#### Get Signature

> **get** **prediction\_upper**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:30](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L30)

Get upper prediction bounds (if requested).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### residuals

#### Get Signature

> **get** **residuals**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:20](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L20)

Get residuals (if requested).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### robustness\_weights

#### Get Signature

> **get** **robustness\_weights**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:32](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L32)

Get robustness weights (if requested).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### standard\_errors

#### Get Signature

> **get** **standard\_errors**(): `Float64Array`\<`ArrayBufferLike`\> \| `null`

Defined in: [index.d.ts:22](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L22)

Get standard errors (if requested/computed).

##### Returns

`Float64Array`\<`ArrayBufferLike`\> \| `null`

***

### x

#### Get Signature

> **get** **x**(): `Float64Array`

Defined in: [index.d.ts:16](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L16)

Get the sorted x values.

##### Returns

`Float64Array`

***

### y

#### Get Signature

> **get** **y**(): `Float64Array`

Defined in: [index.d.ts:18](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L18)

Get the smoothed y values.

##### Returns

`Float64Array`
