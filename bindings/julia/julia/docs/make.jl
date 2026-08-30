using Documenter
using FastLOWESS

const README_PATH = joinpath(@__DIR__, "..", "README.md")
const INDEX_PATH = joinpath(@__DIR__, "src", "index.md")

# Documenter's Markdown parser (unlike GitHub/pkgdown/Starlight/Doxygen) does not
# render the raw `<p align="center">` badge/logo HTML blocks at the top of the
# shared README — convert them to plain Markdown before using it as the homepage.
function html_center_block_to_markdown(block::AbstractString)
	lines = String[]
	for raw_line ∈ eachsplit(strip(block), '\n')
		line = strip(raw_line)
		if (m = match(r"^<a href=\"([^\"]+)\"><img src=\"([^\"]+)\" alt=\"([^\"]+)\"></a>$", line)) !== nothing
			push!(lines, "[![$(m[3])]($(m[2]))]($(m[1]))")
		elseif (m = match(r"^<img src=\"([^\"]+)\" alt=\"([^\"]+)\"(?: width=\"[^\"]+\")?>$", line)) !== nothing
			push!(lines, "![$(m[2])]($(m[1]))")
		elseif (m = match(r"^<em>(.+)</em>$", line)) !== nothing
			push!(lines, "*$(m[1])*")
		elseif line == "<br>" || isempty(line)
			continue
		else
			push!(lines, line)
		end
	end
	return join(lines, "\n\n")
end

function convert_center_block(matched::AbstractString)
	inner = match(r"<p align=\"center\">\n(.*)\n</p>"s, matched).captures[1]
	return html_center_block_to_markdown(inner)
end

readme = read(README_PATH, String)
readme = replace(readme, "\r\n" => "\n")
readme = replace(readme, r"<!--.*?-->\n?"s => "")
readme = replace(readme, r"<p align=\"center\">\n.*?\n</p>"s => convert_center_block)
write(INDEX_PATH, readme)

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
		"Introduction" => ["installation.md", "quickstart.md", "concepts.md"],
		"User Guide" => [
			"adapter-choice.md",
			"batch.md",
			"streaming.md",
			"online.md",
			"intervals.md",
			"cross-validation.md",
		],
		"Weight & Robustness" =>
			["kernels.md", "robustness.md", "scaling.md", "custom-weights.md"],
		"Advanced" => ["boundary.md", "merge.md", "gpu-backend.md"],
		"Use Cases" =>
			["use-case-genomics.md", "use-case-time-series.md", "use-case-real-time.md"],
		"Performance" => ["benchmarks.md"],
		"API Reference" => "api.md",
		"News" => "NEWS.md",
	],
	authors = "Amir Valizadeh",
	warnonly = true,
	checkdocs = :none,
)
