using Documenter
using FastLOWESS

makedocs(
	sitename = "FastLOWESS.jl",
	modules = [FastLOWESS],
	format = Documenter.HTML(
		prettyurls = get(ENV, "CI", "false") == "true",
		canonical = "https://thisisamirv.github.io/lowess-project/julia/stable/",
		repolink = "https://github.com/thisisamirv/lowess-project",
	),
	pages = [
		"Home" => "index.md",
		"Introduction" => [
			"installation.md",
			"quickstart.md",
			"concepts.md",
		],
		"User Guide" => [
			"parameters.md",
			"adapter-choice.md",
			"batch.md",
			"streaming.md",
			"online.md",
			"intervals.md",
			"cross-validation.md",
		],
		"Weight & Robustness" => [
			"kernels.md",
			"robustness.md",
			"scaling.md",
			"custom-weights.md",
		],
		"Advanced" => [
			"boundary.md",
			"merge.md",
			"gpu-backend.md",
		],
		"Use Cases" => [
			"use-case-genomics.md",
			"use-case-time-series.md",
			"use-case-real-time.md",
		],
		"Performance" => [
			"benchmarks.md",
		],
		"API Reference" => "api.md",
	],
	authors = "Amir Valizadeh",
	warnonly = true,
	checkdocs = :none,
)
