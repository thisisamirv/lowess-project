use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

const LAYERS: &[&str] = &[
    "primitives",
    "math",
    "algorithms",
    "evaluation",
    "engine",
    "adapters",
    "api",
];

struct Source {
    path: PathBuf,
    module: Vec<String>,
    layer: usize,
}

fn collect_sources(directory: &Path, root: &Path, sources: &mut Vec<Source>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_sources(&path, root, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let relative = path.strip_prefix(root).unwrap();
            let mut module: Vec<String> = relative
                .iter()
                .map(|part| part.to_str().unwrap().to_owned())
                .collect();
            let stem = path.file_stem().unwrap().to_str().unwrap();
            if stem == "mod" {
                module.pop();
            } else {
                *module.last_mut().unwrap() = stem.to_owned();
            }
            if let Some(layer) = LAYERS.iter().position(|name| *name == module[0]) {
                sources.push(Source {
                    path,
                    module,
                    layer,
                });
            }
        }
    }
}

struct Dependencies<'a> {
    source: &'a Source,
    sources: &'a [Source],
    scope: Vec<String>,
    violations: BTreeSet<String>,
}

impl Dependencies<'_> {
    fn macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        use proc_macro2::TokenTree;
        let mut tokens = tokens.into_iter().peekable();
        while let Some(token) = tokens.next() {
            match token {
                TokenTree::Group(group) => self.macro_tokens(group.stream()),
                TokenTree::Ident(identifier)
                    if matches!(identifier.to_string().as_str(), "crate" | "super" | "self") =>
                {
                    let mut parts = vec![identifier.to_string()];
                    while matches!(tokens.peek(), Some(TokenTree::Punct(punctuation)) if punctuation.as_char() == ':')
                    {
                        tokens.next();
                        if !matches!(tokens.next(), Some(TokenTree::Punct(punctuation)) if punctuation.as_char() == ':')
                        {
                            break;
                        }
                        match tokens.next() {
                            Some(TokenTree::Ident(identifier)) => {
                                parts.push(identifier.to_string())
                            }
                            Some(TokenTree::Group(group)) => {
                                let grouped =
                                    proc_macro2::TokenStream::from(TokenTree::Group(group.clone()));
                                if let Ok(tree) = syn::parse2::<syn::UseTree>(grouped) {
                                    self.use_tree(&tree, parts.clone());
                                } else {
                                    self.macro_tokens(group.stream());
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                    self.check(&parts);
                }
                _ => {}
            }
        }
    }

    fn check(&mut self, parts: &[String]) {
        if parts.is_empty() {
            return;
        }
        let mut resolved = self.scope.clone();
        let mut offset = 0;
        if parts[0] == "crate" {
            resolved.clear();
            offset = 1;
        } else if parts[0] == "self" {
            offset = 1;
        } else {
            while offset < parts.len() && parts[offset] == "super" {
                resolved.pop();
                offset += 1;
            }
        }
        resolved.extend(parts[offset..].iter().cloned());
        let target = self
            .sources
            .iter()
            .filter(|source| resolved.starts_with(&source.module))
            .max_by_key(|source| source.module.len());
        let Some(target) = target else {
            return;
        };
        if target.path == self.source.path {
            return;
        }
        let upward = target.layer > self.source.layer;
        let within_regression = [self.source, target].iter().all(|source| {
            source.module.len() >= 2
                && source.module[0] == "algorithms"
                && source.module[1] == "regression"
        });
        let sibling = target.layer == self.source.layer
            && !within_regression
            && !matches!(
                self.source.path.file_stem().and_then(|stem| stem.to_str()),
                Some("defaults")
            )
            && !matches!(
                target.path.file_stem().and_then(|stem| stem.to_str()),
                Some("defaults" | "errors")
            );
        if upward || sibling {
            self.violations.insert(format!(
                "{} -> {} ({})",
                self.source.module.join("::"),
                target.module.join("::"),
                if upward {
                    "upward"
                } else {
                    "same-layer sibling"
                }
            ));
        }
    }

    fn use_tree(&mut self, tree: &syn::UseTree, prefix: Vec<String>) {
        match tree {
            syn::UseTree::Path(path) => {
                let mut prefix = prefix;
                prefix.push(path.ident.to_string());
                self.use_tree(&path.tree, prefix);
            }
            syn::UseTree::Name(name) => {
                let mut prefix = prefix;
                if name.ident != "self" {
                    prefix.push(name.ident.to_string());
                }
                self.check(&prefix);
            }
            syn::UseTree::Rename(rename) => {
                let mut prefix = prefix;
                prefix.push(rename.ident.to_string());
                self.check(&prefix);
            }
            syn::UseTree::Glob(_) => self.check(&prefix),
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.use_tree(item, prefix.clone());
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for Dependencies<'_> {
    fn visit_macro(&mut self, expression: &'ast syn::Macro) {
        self.visit_path(&expression.path);
        self.macro_tokens(expression.tokens.clone());
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if !matches!(item.vis, syn::Visibility::Inherited) {
            self.violations.insert(format!(
                "{} exposes an implementation-layer forwarding export; import its owner directly",
                self.source.module.join("::")
            ));
        }
        self.use_tree(&item.tree, Vec::new());
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.check(
            &path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>(),
        );
        visit::visit_path(self, path);
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if let Some((_, items)) = &item.content {
            self.scope.push(item.ident.to_string());
            for item in items {
                self.visit_item(item);
            }
            self.scope.pop();
        }
    }
}

fn fixture_sources(entries: &[(&str, &[&str], usize)]) -> Vec<Source> {
    entries
        .iter()
        .map(|(path, module, layer)| Source {
            path: PathBuf::from(path),
            module: module.iter().map(|part| (*part).to_owned()).collect(),
            layer: *layer,
        })
        .collect()
}

#[test]
fn source_dependencies_follow_layers_and_do_not_cross_sibling_files() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    collect_sources(&root, &root, &mut sources);
    let mut violations = BTreeSet::new();
    for source in &sources {
        let ast = syn::parse_file(&fs::read_to_string(&source.path).unwrap()).unwrap();
        let mut visitor = Dependencies {
            source,
            sources: &sources,
            scope: source.module.clone(),
            violations: BTreeSet::new(),
        };
        visitor.visit_file(&ast);
        violations.extend(visitor.violations);
    }
    assert!(
        violations.is_empty(),
        "Dependency rules violated:\n{}",
        violations.into_iter().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn checker_enforces_direction_and_sibling_exceptions() {
    let sources = fixture_sources(&[
        ("engine/executor.rs", &["engine", "executor"], 4),
        ("engine/validator.rs", &["engine", "validator"], 4),
        ("engine/defaults.rs", &["engine", "defaults"], 4),
        ("engine/errors.rs", &["engine", "errors"], 4),
        ("evaluation/intervals.rs", &["evaluation", "intervals"], 3),
        ("adapters/predict.rs", &["adapters", "predict"], 5),
    ]);
    let mut visitor = Dependencies {
        source: &sources[0],
        sources: &sources,
        scope: sources[0].module.clone(),
        violations: BTreeSet::new(),
    };
    visitor.visit_file(&syn::parse_file("use crate::evaluation::intervals::IntervalMethod; use crate::engine::{defaults::DEFAULT_FRACTION, errors::Error};").unwrap());
    assert!(visitor.violations.is_empty());
    visitor.visit_file(
        &syn::parse_file(
            "use crate::engine::validator::Validator; use crate::adapters::predict::Predict;",
        )
        .unwrap(),
    );
    assert_eq!(visitor.violations.len(), 2);
    assert!(
        visitor
            .violations
            .iter()
            .any(|violation| violation.contains("upward"))
    );
    assert!(
        visitor
            .violations
            .iter()
            .any(|violation| violation.contains("same-layer sibling"))
    );
    visitor.violations.clear();
    visitor.visit_file(
        &syn::parse_file("macro_rules! forbidden { () => { crate::adapters::predict::Predict }; }")
            .unwrap(),
    );
    assert_eq!(visitor.violations.len(), 1);
    assert!(
        visitor
            .violations
            .iter()
            .all(|violation| violation.contains("upward"))
    );
}

#[test]
fn checker_allows_only_internal_regression_dependencies() {
    let sources = fixture_sources(&[
        (
            "algorithms/regression/context.rs",
            &["algorithms", "regression", "context"],
            2,
        ),
        (
            "algorithms/regression/generic.rs",
            &["algorithms", "regression", "generic"],
            2,
        ),
        (
            "algorithms/regression/specialized/mod.rs",
            &["algorithms", "regression", "specialized"],
            2,
        ),
        (
            "algorithms/interpolation.rs",
            &["algorithms", "interpolation"],
            2,
        ),
        ("engine/executor.rs", &["engine", "executor"], 4),
    ]);
    let mut visitor = Dependencies {
        source: &sources[0],
        sources: &sources,
        scope: sources[0].module.clone(),
        violations: BTreeSet::new(),
    };
    visitor.visit_file(
        &syn::parse_file(
            "use super::generic::Solver; use crate::algorithms::regression::specialized::Solver;",
        )
        .unwrap(),
    );
    assert!(visitor.violations.is_empty());
    visitor.visit_file(&syn::parse_file("use crate::algorithms::interpolation::Interpolation; use crate::engine::executor::Executor;").unwrap());
    assert_eq!(visitor.violations.len(), 2);
    let mut outside = Dependencies {
        source: &sources[3],
        sources: &sources,
        scope: sources[3].module.clone(),
        violations: BTreeSet::new(),
    };
    outside.visit_file(
        &syn::parse_file("use crate::algorithms::regression::generic::Solver;").unwrap(),
    );
    assert_eq!(outside.violations.len(), 1);
}

#[test]
fn checker_rejects_forwarding_exports_even_when_the_owner_is_lower() {
    let sources = fixture_sources(&[
        ("engine/executor.rs", &["engine", "executor"], 4),
        ("evaluation/intervals.rs", &["evaluation", "intervals"], 3),
    ]);
    let mut visitor = Dependencies {
        source: &sources[0],
        sources: &sources,
        scope: sources[0].module.clone(),
        violations: BTreeSet::new(),
    };
    visitor
        .visit_file(&syn::parse_file("use crate::evaluation::intervals::IntervalMethod;").unwrap());
    assert!(visitor.violations.is_empty());
    visitor.visit_file(
        &syn::parse_file("pub use crate::evaluation::intervals::IntervalMethod;").unwrap(),
    );
    assert_eq!(visitor.violations.len(), 1);
    assert!(
        visitor
            .violations
            .iter()
            .all(|violation| violation.contains("forwarding export"))
    );
}

#[test]
fn checker_allows_defaults_siblings_but_not_upward_imports() {
    let sources = fixture_sources(&[
        ("math/defaults.rs", &["math", "defaults"], 1),
        ("math/kernel.rs", &["math", "kernel"], 1),
        ("primitives/backend.rs", &["primitives", "backend"], 0),
        ("primitives/policies.rs", &["primitives", "policies"], 0),
        ("evaluation/intervals.rs", &["evaluation", "intervals"], 3),
    ]);
    let mut defaults = Dependencies {
        source: &sources[0],
        sources: &sources,
        scope: sources[0].module.clone(),
        violations: BTreeSet::new(),
    };
    defaults.visit_file(&syn::parse_file("use crate::math::kernel::WeightFunction;").unwrap());
    assert!(defaults.violations.is_empty());
    defaults
        .visit_file(&syn::parse_file("use crate::evaluation::intervals::IntervalMethod;").unwrap());
    assert_eq!(defaults.violations.len(), 1);
    assert!(
        defaults
            .violations
            .iter()
            .all(|violation| violation.contains("upward"))
    );
    let mut primitive = Dependencies {
        source: &sources[2],
        sources: &sources,
        scope: sources[2].module.clone(),
        violations: BTreeSet::new(),
    };
    primitive
        .visit_file(&syn::parse_file("use crate::primitives::policies::MissingPolicy;").unwrap());
    assert_eq!(primitive.violations.len(), 1);
    assert!(
        primitive
            .violations
            .iter()
            .all(|violation| violation.contains("same-layer sibling"))
    );
    primitive.violations.clear();
    primitive.visit_file(&syn::parse_file("use crate::math::kernel::WeightFunction;").unwrap());
    assert_eq!(primitive.violations.len(), 1);
    assert!(
        primitive
            .violations
            .iter()
            .all(|violation| violation.contains("upward"))
    );
}
