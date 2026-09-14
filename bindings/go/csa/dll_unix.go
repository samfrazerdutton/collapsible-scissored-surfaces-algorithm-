//go:build !windows

// The Linux/macOS dynamic-loading plumbing, mirroring dll_windows.go's
// small surface (lazyProc, mustLoadLibrary, envOr, fileExists) -- but NOT
// its bufferReturningCall, because the two platforms disagree on how a
// small struct is returned by value:
//
//   - Microsoft x64 ABI (Windows, csa.dll built with MSVC): a 16-byte
//     struct return is always passed back via a hidden out-pointer
//     supplied as an implicit first argument (see dll_windows.go's own
//     bufferReturningCall).
//   - System V AMD64 ABI (Linux) and the equivalent Darwin AMD64/ARM64
//     ABI: a two-eightbyte, all-INTEGER-class struct like
//     `csa_buffer{void* data; size_t size;}` is instead packed directly
//     into two return registers (RAX:RDX) -- no hidden pointer at all.
//
// purego.SyscallN exposes both registers as (r1, r2), which is exactly
// what's needed to reconstruct the struct correctly here.
//
// IMPORTANT PROVENANCE NOTE: this file was written and cross-compiled
// (GOOS=linux and GOOS=darwin, both amd64 and arm64) to confirm it type-
// checks, but this development environment is Windows-only -- there is no
// Linux or macOS machine available to actually load a real libcsa.so/
// .dylib and run the existing csa_test.go suite against it. Every other
// claim in this codebase is backed by a real measured test run; this file
// is the one exception, so treat it as reviewed-and-believed-correct
// (the ABI reasoning above is the standard, well-documented one for both
// platforms) rather than empirically verified the way everything else
// here is.
package csa

import (
	"os"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

var defaultLibraryName = func() string {
	if runtime.GOOS == "darwin" {
		return "libcsa.dylib"
	}
	return "libcsa.so"
}()

type lazyProc struct {
	addr uintptr
}

func newLazyProc(handle uintptr, name string) *lazyProc {
	addr, err := purego.Dlsym(handle, name)
	if err != nil {
		panic("csa: could not find exported symbol " + name + " in " + defaultLibraryName + ": " + err.Error())
	}
	return &lazyProc{addr: addr}
}

// call mirrors dll_windows.go's lazyProc.call: only the first return
// register (r1/RAX) is meaningful for every plain-scalar-returning
// function in csa_capi.h (csa_last_error, csa_free_buffer,
// csa_cuda_available); csa_buffer-returning functions go through
// bufferReturningCall below instead, which needs both r1 and r2.
func (p *lazyProc) call(args ...uintptr) uintptr {
	r1, _, _ := purego.SyscallN(p.addr, args...)
	return r1
}

func bufferReturningCall(p *lazyProc, args ...uintptr) csaBuffer {
	r1, r2, _ := purego.SyscallN(p.addr, args...)
	// r1 is a raw address handed back directly in a CPU register by the
	// callee, not derived from a Go unsafe.Pointer -- the same provenance
	// goStringFromCStr's own uintptr-to-Pointer conversion in csa.go
	// relies on, for the same reason.
	return csaBuffer{data: unsafe.Pointer(r1), size: r2}
}

func mustLoadLibrary() uintptr {
	path, err := libraryPath()
	if err != nil {
		panic(err)
	}
	handle, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		panic("csa: failed to load " + path + ": " + err.Error())
	}
	return handle
}

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
