[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / StreamingOptions

# Interface: StreamingOptions

Defined in: [index.d.ts:160](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L160)

Configuration options for streaming processing.

## Properties

### chunk\_size?

> `optional` **chunk\_size?**: `number`

Defined in: [index.d.ts:162](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L162)

Size of each data chunk. Default: 5000.

***

### merge\_strategy?

> `optional` **merge\_strategy?**: `string`

Defined in: [index.d.ts:166](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L166)

Strategy for merging chunk overlaps. Default: "weighted_average".

***

### overlap?

> `optional` **overlap?**: `number`

Defined in: [index.d.ts:164](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L164)

Header/footer overlap size. Default: chunk_size / 10, min. 1.
