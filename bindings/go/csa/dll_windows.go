//go:build windows

// The actual Windows dynamic-loading plumbing, isolated from csa.go so
// the platform-specific bits (syscall.DLL/Proc) are all in one place --
// a future non-Windows build (cgo or purego against .so/.dylib) would
// only need a new file with this same small surface (lazyProc,
// mustLoadLibrary, envOr, fileExists), not changes to csa.go itself.
package csa

import (
	"os"
	"syscall"
	"unsafe"
)

const defaultLibraryName = "csa.dll"

type lazyProc struct {
	proc *syscall.Proc
}

func newLazyProc(dll *syscall.DLL, name string) *lazyProc {
	p, err := dll.FindProc(name)
	if err != nil {
		panic("csa: could not find exported symbol " + name + " in csa.dll: " + err.Error())
	}
	return &lazyProc{proc: p}
}

// call ignores Proc.Call's error return by design: syscall.Proc.Call's
// third return value is always non-nil (populated from GetLastError()
// regardless of whether the call actually failed), and none of libcsa's
// C ABI functions set a Windows last-error code -- checking it here would
// just be treating GetLastError() noise as a real failure signal, which
// it isn't for this DLL's contract (errors are reported via
// csa_last_error(), not the Windows error code).
func (p *lazyProc) call(args ...uintptr) uintptr {
	r1, _, _ := p.proc.Call(args...)
	return r1
}

// bufferReturningCall implements the Microsoft x64 ABI's hidden-out-
// pointer convention for a 16-byte-by-value struct return (see dll_unix.go's
// own bufferReturningCall for why Linux/macOS need a different
// implementation instead of sharing this one).
func bufferReturningCall(p *lazyProc, args ...uintptr) csaBuffer {
	var buf csaBuffer
	full := make([]uintptr, 0, len(args)+1)
	full = append(full, uintptr(unsafe.Pointer(&buf)))
	full = append(full, args...)
	p.call(full...)
	return buf
}

func mustLoadLibrary() *syscall.DLL {
	path, err := libraryPath()
	if err != nil {
		panic(err)
	}
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		panic("csa: failed to load " + path + ": " + err.Error())
	}
	return dll
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
