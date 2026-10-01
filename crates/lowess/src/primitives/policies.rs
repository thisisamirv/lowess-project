//! Shared input and update policies for LOWESS execution.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingPolicy {
    #[default]
    Error,
    Drop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    Full,
    #[default]
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    Average,
    #[default]
    WeightedAverage,
    TakeFirst,
    TakeLast,
}
