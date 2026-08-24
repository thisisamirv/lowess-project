[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / OnlineOutput

# Interface: OnlineOutput

Defined in: [index.d.ts:93](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L93)

Result of a single online update step.

## Properties

### iterations\_used?

> `optional` **iterations\_used?**: `number`

Defined in: [index.d.ts:103](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L103)

Number of robustness iterations performed (if applicable).

***

### residual?

> `optional` **residual?**: `number`

Defined in: [index.d.ts:99](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L99)

Residual (raw input y minus this output's y) (if computed).

***

### robustness\_weight?

> `optional` **robustness\_weight?**: `number`

Defined in: [index.d.ts:101](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L101)

Robustness weight for the latest point (if computed).

***

### standard\_error?

> `optional` **standard\_error?**: `number`

Defined in: [index.d.ts:97](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L97)

Standard error (if computed).

***

### y

> **y**: `number`

Defined in: [index.d.ts:95](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L95)

Smoothed value for the latest point.
