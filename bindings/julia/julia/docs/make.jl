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
		if (
			m = match(
				r"^<a href=\"([^\"]+)\"><img src=\"([^\"]+)\" alt=\"([^\"]+)\"></a>$",
				line,
			)
		) !== nothing
			push!(lines, "[![$(m[3])]($(m[2]))]($(m[1]))")
		elseif (
			m = match(
				r"^<img src=\"([^\"]+)\" alt=\"([^\"]+)\"(?: width=\"[^\"]+\")?>$",
				line,
			)
		) !== nothing
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
readme = replace(readme, r"\A<!--.*?-->\n?"s => "")
readme = replace(readme, r"<p align=\"center\">\n.*?\n</p>"s => convert_center_block)
# index.md is its own generated page with its own markdownlint needs (the
# tagline right after the logo trips MD036), not the same as README.md's.
readme = "<!-- markdownlint-disable MD036 -->\n" * readme
write(INDEX_PATH, readme)

# Documenter's Markdown parser also does not treat `<!-- ... -->` as an invisible
# HTML comment — it renders it as literal text (e.g. a leading
# `<!-- markdownlint-disable ... -->` shows up verbatim on the page). Every
# docs/src/*.md page keeps such a comment on its first line for editor/markdownlint
# purposes, so temporarily strip just that leading comment line from each page
# before building, then restore the original file afterwards — this keeps the
# comment in the on-disk source while hiding it from the rendered site.
const SRC_DIR = joinpath(@__DIR__, "src")
const LEADING_COMMENT_RE = r"\A<!--.*?-->\n?"s

original_contents = Dict{String, String}()
md_paths = String[]
for (root, _, files) ∈ walkdir(SRC_DIR)
	for file ∈ files
		endswith(file, ".md") && push!(md_paths, joinpath(root, file))
	end
end
for path ∈ md_paths
	content = read(path, String)
	if occursin(LEADING_COMMENT_RE, content)
		original_contents[path] = content
		write(path, replace(content, LEADING_COMMENT_RE => ""))
	end
end

try
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
				"introduction/installation.md",
				"introduction/quickstart.md",
				"introduction/concepts.md",
			],
			"User Guide" => [
				"guide/adapter-choice.md",
				"guide/intervals.md",
				"guide/cross-validation.md",
			],
			"Weight & Robustness" => [
				"weighting/kernels.md",
				"weighting/robustness.md",
				"weighting/scaling.md",
				"weighting/custom-weights.md",
			],
			"Advanced" =>
				["advanced/boundary.md", "advanced/merge.md", "advanced/gpu-backend.md"],
			"Use Cases" => [
				"use-case/use-case-genomics.md",
				"use-case/use-case-time-series.md",
				"use-case/use-case-real-time.md",
			],
			"Performance" => ["benchmarks.md"],
			"API Guide" => ["api/api.md", "api/api-streaming.md", "api/api-online.md"],
			"API Reference" => "api.md",
			"News" => "NEWS.md",
		],
		authors = "Amir Valizadeh",
		warnonly = true,
		checkdocs = :none,
	)
finally
	for (path, content) ∈ original_contents
		write(path, content)
	end
end
