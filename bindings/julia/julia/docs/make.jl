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
		"API Reference" => "api.md",
	],
	authors = "Amir Valizadeh",
	warnonly = true,
	checkdocs = :none,
)
