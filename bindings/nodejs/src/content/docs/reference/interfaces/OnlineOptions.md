[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / OnlineOptions

# Interface: OnlineOptions

Defined in: [index.d.ts:83](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L83)

Configuration options for online processing.

## Properties

### min\_points?

> `optional` **min\_points?**: `number`

Defined in: [index.d.ts:87](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L87)

Minimum points required before smoothing starts. Default: 3.

***

### update\_mode?

> `optional` **update\_mode?**: `string`

Defined in: [index.d.ts:89](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L89)

Update mode ("full", "incremental"). Default: "full".

***

### window\_capacity?

> `optional` **window\_capacity?**: `number`

Defined in: [index.d.ts:85](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L85)

Maximum number of points to keep in the window. Default: 1000.
