package fastlowess

// One-time downloader for the opt-in GPU-enabled fastlowess_go static
// library. Unlike Python/Node.js/R/C++/Julia, cgo links this library
// statically at build time, so there's no runtime install: this only
// downloads the library, then instructs the caller to rebuild with
// CGO_LDFLAGS pointing at it.

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const gpuRepo = "thisisamirv/lowess-project"

// GPU artifacts across all versions live in this one perpetual release
// instead of cluttering each version's own release page; the source version
// is embedded in each asset's filename instead.
const gpuReleaseTag = "gpu-builds"
const gpuArchiveMarkerPrefix = "fastlowess-go-gpu|abi-v4|"

func gpuBuildMarker() string {
	tag := platformArchTag()
	if tag == "" {
		return ""
	}
	if runtime.GOOS == "linux" {
		tag += "-glibc"
	}
	return gpuArchiveMarkerPrefix + tag
}

// platformArchTag returns the "<platform>-<arch>" suffix release-gpu.yml
// uses for this binding's GPU static library, or "" if this platform/arch
// combination isn't built.
func platformArchTag() string {
	switch runtime.GOOS {
	case "linux":
		if !nativeUsesGlibc() {
			return ""
		}
		switch runtime.GOARCH {
		case "amd64":
			return "linux-x86_64"
		case "arm64":
			return "linux-arm64"
		}
	case "darwin":
		switch runtime.GOARCH {
		case "amd64":
			return "macos-x86_64"
		case "arm64":
			return "macos-aarch64"
		}
	case "windows":
		switch runtime.GOARCH {
		case "amd64":
			return "windows-x86_64"
		case "arm64":
			return "windows-arm64"
		}
	}
	return ""
}

// InstallGPU installs a GPU-enabled fastlowess_go static library for this
// platform, saving it to ~/.fastlowess/gpu/libfastlowess_go.a.
//
// Pass yes=true to skip the interactive y/N confirmation prompt; this is
// required when stdin is not an interactive terminal.
//
// Pass a non-empty localPath to install a GPU-enabled library already built
// locally (e.g. via `cargo build -p fastlowess-go --profile release-c
// --features gpu`) instead of downloading one from the matching GitHub
// Release — useful for testing the installer itself, or installing an
// unreleased build.
func InstallGPU(yes bool, localPath string) error {
	if GPUEnabled() {
		fmt.Println("GPU backend is already active.")
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	dir := filepath.Join(home, ".fastlowess", "gpu")
	dest := filepath.Join(dir, "libfastlowess_go.a")

	if localPath != "" {
		if _, err := os.Stat(localPath); err != nil {
			return fmt.Errorf("no such file: %s", localPath)
		}
		marker := gpuBuildMarker()
		if marker == "" {
			return fmt.Errorf("no matching GPU build target for %s/%s", runtime.GOOS, runtime.GOARCH)
		}
		if !yes {
			fmt.Printf("Install %s in place of the current build? [y/N] ", localPath)
			reader := bufio.NewReader(os.Stdin)
			answer, _ := reader.ReadString('\n')
			answer = strings.TrimSpace(strings.ToLower(answer))
			if answer != "y" && answer != "yes" {
				fmt.Println("Aborted.")
				return nil
			}
		}

		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("failed to create %s: %w", dir, err)
		}
		fmt.Printf("Installing %s ...\n", localPath)
		if err := copyGPULibrary(localPath, dest, marker); err != nil {
			return fmt.Errorf("failed to copy %s: %w", localPath, err)
		}
		printRebuildInstructions(dest, dir)
		return nil
	}

	tag := platformArchTag()
	if tag == "" {
		return fmt.Errorf("no prebuilt GPU library available for %s/%s; build from source instead: "+
			"cargo build -p fastlowess-go --profile release-c --features gpu", runtime.GOOS, runtime.GOARCH)
	}

	assetName := fmt.Sprintf("libfastlowess_go-gpu-v%s-%s.a", version, tag)
	url := fmt.Sprintf("https://github.com/%s/releases/download/%s/%s", gpuRepo, gpuReleaseTag, assetName)

	if !yes {
		fmt.Printf("Download and install %s from github.com/%s? [y/N] ", assetName, gpuRepo)
		reader := bufio.NewReader(os.Stdin)
		answer, _ := reader.ReadString('\n')
		answer = strings.TrimSpace(strings.ToLower(answer))
		if answer != "y" && answer != "yes" {
			fmt.Println("Aborted.")
			return nil
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("failed to create %s: %w", dir, err)
	}

	fmt.Printf("Downloading %s ...\n", url)
	if err := downloadGPULibrary(url, dest, gpuBuildMarker()); err != nil {
		return fmt.Errorf("failed to download %s: %w\n"+
			"a matching GPU build may not exist for this platform/version yet", url, err)
	}

	printRebuildInstructions(dest, dir)
	return nil
}

func printRebuildInstructions(dest, dir string) {
	fmt.Printf("GPU library installed at %s.\n", dest)
	fmt.Println("A running program can't swap a statically-linked library, so rebuild with " +
		"CGO_LDFLAGS pointing at it, e.g.:")
	switch runtime.GOOS {
	case "windows":
		fmt.Printf("  set CGO_LDFLAGS=-L%s -lfastlowess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread\n", dir)
	case "darwin":
		fmt.Printf("  export CGO_LDFLAGS=\"-L%s -lfastlowess_go\"\n", dir)
	default:
		fmt.Printf("  export CGO_LDFLAGS=\"-L%s -lfastlowess_go -lm -ldl -lpthread\"\n", dir)
	}
	fmt.Println("then `go build -tags=external_native ./...`.")
}

func copyGPULibrary(src, dest, marker string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()

	tmp := dest + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := validateGPUArchive(tmp, marker); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dest)
}

func downloadGPULibrary(url, dest, marker string) error {
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := validateGPUArchive(tmp, marker); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dest)
}

func validateGPUArchive(path, marker string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()

	var magic [8]byte
	if _, err := io.ReadFull(file, magic[:]); err != nil {
		return fmt.Errorf("invalid static library archive: %w", err)
	}
	if string(magic[:]) != "!<arch>\n" {
		return fmt.Errorf("invalid static library archive: missing ar signature")
	}

	needle := []byte(marker)
	chunk := make([]byte, 32*1024)
	tail := make([]byte, 0, len(needle)-1)
	for {
		n, readErr := file.Read(chunk)
		data := append(tail, chunk[:n]...)
		if bytes.Contains(data, needle) {
			return nil
		}
		keep := len(needle) - 1
		if keep > len(data) {
			keep = len(data)
		}
		tail = append(tail[:0], data[len(data)-keep:]...)
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return readErr
		}
	}
	return fmt.Errorf("archive does not contain GPU build marker %q", marker)
}
