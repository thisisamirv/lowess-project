[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / OnlineLowess

# Class: OnlineLowess

Defined in: [index.d.ts:44](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L44)

Online LOWESS smoother for real-time data.

## Constructors

### Constructor

> **new OnlineLowess**(`options?`, `onlineOpts?`): `OnlineLowess`

Defined in: [index.d.ts:46](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L46)

Create a new online LOWESS smoother.

#### Parameters

##### options?

[`SmoothOptions`](../interfaces/SmoothOptions.md) \| `null`

##### onlineOpts?

[`OnlineOptions`](../interfaces/OnlineOptions.md) \| `null`

#### Returns

`OnlineLowess`

## Methods

### add\_point()

> **add\_point**(`x`, `y`): [`OnlineOutput`](../interfaces/OnlineOutput.md) \| `null`

Defined in: [index.d.ts:48](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L48)

Add a single point and get the smoothed value if enough points are available.

#### Parameters

##### x

`number`

##### y

`number`

#### Returns

[`OnlineOutput`](../interfaces/OnlineOutput.md) \| `null`
