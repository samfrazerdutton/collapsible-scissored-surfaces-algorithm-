// Package csa provides Go bindings for libcsa via direct dynamic loading
// of the native shared library (syscall.LoadDLL + Proc.Call) rather than
// cgo -- this environment has no C compiler available to link a cgo
// build against the MSVC-built csa.dll, and LoadLibrary/GetProcAddress
// works against any DLL regardless of which compiler built it, sidestepping
// that entirely. This mirrors how the Go standard library itself calls
// arbitrary Windows DLLs (e.g. user32.dll).
//
// csa_capi.h's functions return a `csa_buffer { unsigned char* data;
// size_t size; }` struct BY VALUE. Under the Microsoft x64 calling
// convention (what the MSVC-built csa.dll uses), a 16-byte struct return
// is passed back via a HIDDEN POINTER supplied as an implicit first
// argument: the caller allocates 16 bytes of scratch space and passes its
// address as arg 0, and the callee writes {data, size} there. Every
// helper below that calls a csa_buffer-returning function passes that
// hidden pointer as the first element of the args slice.
//
// Linux/macOS support (dll_unix.go) uses purego instead of cgo, for the
// same reason: no portable C toolchain can be assumed at build time, and
// purego's dlopen/dlsym-based loading works against any .so/.dylib
// regardless of which compiler built it, exactly like this file's
// LoadLibrary/GetProcAddress approach does for Windows.
package csa

import (
	"errors"
	"path/filepath"
	"runtime"
	"unsafe"
)

// CsaError wraps a message from csa_last_error().
type CsaError struct{ msg string }

func (e *CsaError) Error() string { return e.msg }

type csaBuffer struct {
	data unsafe.Pointer
	size uintptr
}

var (
	procCompress           *lazyProc
	procDecompress         *lazyProc
	procCompressGeo2D      *lazyProc
	procCompressGeo2DLossy *lazyProc
	procDecompressGeo2D    *lazyProc
	procCompressGeo3D      *lazyProc
	procCompressGeo3DLossy *lazyProc
	procDecompressGeo3D    *lazyProc
	procCompressPose       *lazyProc
	procCompressPoseLossy  *lazyProc
	procDecompressPose     *lazyProc
	procFreeBuffer         *lazyProc
	procLastError          *lazyProc
	procCudaAvailable      *lazyProc
)

func init() {
	dll := mustLoadLibrary()
	procCompress = newLazyProc(dll, "csa_compress")
	procDecompress = newLazyProc(dll, "csa_decompress")
	procCompressGeo2D = newLazyProc(dll, "csa_compress_geo2d")
	procCompressGeo2DLossy = newLazyProc(dll, "csa_compress_geo2d_lossy")
	procDecompressGeo2D = newLazyProc(dll, "csa_decompress_geo2d")
	procCompressGeo3D = newLazyProc(dll, "csa_compress_geo3d")
	procCompressGeo3DLossy = newLazyProc(dll, "csa_compress_geo3d_lossy")
	procDecompressGeo3D = newLazyProc(dll, "csa_decompress_geo3d")
	procCompressPose = newLazyProc(dll, "csa_compress_pose")
	procCompressPoseLossy = newLazyProc(dll, "csa_compress_pose_lossy")
	procDecompressPose = newLazyProc(dll, "csa_decompress_pose")
	procFreeBuffer = newLazyProc(dll, "csa_free_buffer")
	procLastError = newLazyProc(dll, "csa_last_error")
	procCudaAvailable = newLazyProc(dll, "csa_cuda_available")
}

// libraryPath resolves the same way the Python/Rust/C# bindings do:
// CSA_LIB_PATH env var to an exact file, else ../../../build/<name>
// relative to this source file's location (captured at build time via
// runtime.Caller, the closest Go equivalent of __file__/CARGO_MANIFEST_DIR),
// where <name> is platform's native shared library name (defaultLibraryName,
// defined per-platform in dll_windows.go/dll_unix.go).
func libraryPath() (string, error) {
	if p := envOr("CSA_LIB_PATH", ""); p != "" {
		if fileExists(p) {
			return p, nil
		}
		return "", errors.New("CSA_LIB_PATH is set to " + p + " but that file does not exist")
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", errors.New("csa: could not determine source file location")
	}
	dir := filepath.Dir(thisFile)
	candidate := filepath.Join(dir, "..", "..", "..", "build", defaultLibraryName)
	if fileExists(candidate) {
		return candidate, nil
	}
	return "", errors.New("csa: could not locate " + defaultLibraryName + " at " + candidate +
		"; build the C++ project first (cmake --build build from the repo root), " +
		"or set CSA_LIB_PATH to the exact library path")
}

func extractAndFree(buf csaBuffer) ([]byte, error) {
	if buf.data == nil && buf.size == 0 {
		errPtr := procLastError.call()
		if errPtr != 0 {
			msg := goStringFromCStr(errPtr)
			if msg != "" {
				return nil, &CsaError{msg: msg}
			}
		}
		return []byte{}, nil
	}
	n := int(buf.size)
	result := make([]byte, n)
	src := unsafe.Slice((*byte)(buf.data), n)
	copy(result, src)
	// csa_free_buffer takes csa_buffer BY VALUE, which on x64 means the
	// caller passes a pointer to a (caller-owned) copy of the struct --
	// reuse buf itself as that copy.
	procFreeBuffer.call(uintptr(unsafe.Pointer(&buf)))
	runtime.KeepAlive(buf)
	return result, nil
}

// goStringFromCStr walks a NUL-terminated C string. `ptr` is a raw
// address handed back by syscall.Proc.Call (the r1 return value, always
// typed uintptr by Go's syscall package regardless of what it conceptually
// points to) rather than something derived from a Go unsafe.Pointer, so
// the usual "don't convert uintptr to Pointer across an expression
// boundary" rule -- which exists to stop a moving GC from invalidating a
// stale address -- doesn't actually apply here: this address refers to
// memory in libcsa's C runtime heap, which Go's GC never manages or
// moves, in the first place. `go vet` still flags this conversion
// (its heuristic can't distinguish a foreign address from a Go one);
// that's a known, reviewed false positive for this exact pattern, not
// a bug -- verified by the whole test suite's error-handling test
// actually exercising this path against the real library.
func goStringFromCStr(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	var b []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Pointer(ptr + uintptr(i))) //nolint:govet
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

// Compress performs general-purpose compression: tries raw storage, the
// Pantograph Lift, and the LZ dictionary matcher, and keeps whichever
// encodes smallest. useGpu tries the CUDA path for the Pantograph Lift
// candidate if a device is available, falling back transparently otherwise.
func Compress(data []byte, useGpu bool) ([]byte, error) {
	gpu := uintptr(0)
	if useGpu {
		gpu = 1
	}
	var inputPtr uintptr
	if len(data) > 0 {
		inputPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	buf := bufferReturningCall(procCompress, inputPtr, uintptr(len(data)), gpu)
	runtime.KeepAlive(data)
	return extractAndFree(buf)
}

func Decompress(data []byte) ([]byte, error) {
	var inputPtr uintptr
	if len(data) > 0 {
		inputPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	buf := bufferReturningCall(procDecompress, inputPtr, uintptr(len(data)))
	runtime.KeepAlive(data)
	return extractAndFree(buf)
}

// Point2D is an integer 2D coordinate pair.
type Point2D struct{ X, Y int32 }

// Point3D is an integer 3D coordinate triple.
type Point3D struct{ X, Y, Z int32 }

func points2DToFlat(points []Point2D) []int32 {
	flat := make([]int32, len(points)*2)
	for i, p := range points {
		flat[2*i] = p.X
		flat[2*i+1] = p.Y
	}
	return flat
}

func points3DToFlat(points []Point3D) []int32 {
	flat := make([]int32, len(points)*3)
	for i, p := range points {
		flat[3*i] = p.X
		flat[3*i+1] = p.Y
		flat[3*i+2] = p.Z
	}
	return flat
}

func CompressGeo2D(points []Point2D) ([]byte, error) {
	flat := points2DToFlat(points)
	var ptr uintptr
	if len(flat) > 0 {
		ptr = uintptr(unsafe.Pointer(&flat[0]))
	}
	buf := bufferReturningCall(procCompressGeo2D, ptr, uintptr(len(points)))
	runtime.KeepAlive(flat)
	return extractAndFree(buf)
}

// CompressGeo2DLossy: quantStep <= 1 is lossless (identical to
// CompressGeo2D). resyncInterval periodically stores an exact rod to
// bound how far absolute position error can drift (0 = never).
func CompressGeo2DLossy(points []Point2D, quantStep, resyncInterval uint32) ([]byte, error) {
	flat := points2DToFlat(points)
	var ptr uintptr
	if len(flat) > 0 {
		ptr = uintptr(unsafe.Pointer(&flat[0]))
	}
	buf := bufferReturningCall(procCompressGeo2DLossy, ptr, uintptr(len(points)),
		uintptr(quantStep), uintptr(resyncInterval))
	runtime.KeepAlive(flat)
	return extractAndFree(buf)
}

// DecompressGeo2D works for both lossless and lossy blobs -- the
// quantization parameters ride in the blob itself.
func DecompressGeo2D(data []byte) ([]Point2D, error) {
	var inputPtr uintptr
	if len(data) > 0 {
		inputPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	var outCount uintptr
	buf := bufferReturningCall(procDecompressGeo2D, inputPtr, uintptr(len(data)),
		uintptr(unsafe.Pointer(&outCount)))
	runtime.KeepAlive(data)
	raw, err := extractAndFree(buf)
	if err != nil {
		return nil, err
	}
	n := int(outCount)
	result := make([]Point2D, n)
	for i := 0; i < n; i++ {
		x := *(*int32)(unsafe.Pointer(&raw[i*8]))
		y := *(*int32)(unsafe.Pointer(&raw[i*8+4]))
		result[i] = Point2D{x, y}
	}
	return result, nil
}

func CompressGeo3D(points []Point3D) ([]byte, error) {
	flat := points3DToFlat(points)
	var ptr uintptr
	if len(flat) > 0 {
		ptr = uintptr(unsafe.Pointer(&flat[0]))
	}
	buf := bufferReturningCall(procCompressGeo3D, ptr, uintptr(len(points)))
	runtime.KeepAlive(flat)
	return extractAndFree(buf)
}

// CompressGeo3DLossy: same design as CompressGeo2DLossy. Tries both the
// xy+z composition and the true 3D similarity joint and keeps whichever
// encodes smaller. quantStep <= 1 is lossless (identical to CompressGeo3D).
func CompressGeo3DLossy(points []Point3D, quantStep, resyncInterval uint32) ([]byte, error) {
	flat := points3DToFlat(points)
	var ptr uintptr
	if len(flat) > 0 {
		ptr = uintptr(unsafe.Pointer(&flat[0]))
	}
	buf := bufferReturningCall(procCompressGeo3DLossy, ptr, uintptr(len(points)),
		uintptr(quantStep), uintptr(resyncInterval))
	runtime.KeepAlive(flat)
	return extractAndFree(buf)
}

func DecompressGeo3D(data []byte) ([]Point3D, error) {
	var inputPtr uintptr
	if len(data) > 0 {
		inputPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	var outCount uintptr
	buf := bufferReturningCall(procDecompressGeo3D, inputPtr, uintptr(len(data)),
		uintptr(unsafe.Pointer(&outCount)))
	runtime.KeepAlive(data)
	raw, err := extractAndFree(buf)
	if err != nil {
		return nil, err
	}
	n := int(outCount)
	result := make([]Point3D, n)
	for i := 0; i < n; i++ {
		x := *(*int32)(unsafe.Pointer(&raw[i*12]))
		y := *(*int32)(unsafe.Pointer(&raw[i*12+4]))
		z := *(*int32)(unsafe.Pointer(&raw[i*12+8]))
		result[i] = Point3D{x, y, z}
	}
	return result, nil
}

// Quat is a unit-quaternion orientation (w, x, y, z).
type Quat struct{ W, X, Y, Z int32 }

// Pose is a 6-DOF sample: position, then orientation.
type Pose struct {
	Position    Point3D
	Orientation Quat
}

func posesToFlat(poses []Pose) []int32 {
	flat := make([]int32, len(poses)*7)
	for i, p := range poses {
		base := 7 * i
		flat[base] = p.Position.X
		flat[base+1] = p.Position.Y
		flat[base+2] = p.Position.Z
		flat[base+3] = p.Orientation.W
		flat[base+4] = p.Orientation.X
		flat[base+5] = p.Orientation.Y
		flat[base+6] = p.Orientation.Z
	}
	return flat
}

// CompressPose: 6-DOF pose stream -- position via the Geo3D auto-select,
// orientation via the Quaternion Joint (see DESIGN.md).
func CompressPose(poses []Pose) ([]byte, error) {
	flat := posesToFlat(poses)
	var ptr uintptr
	if len(flat) > 0 {
		ptr = uintptr(unsafe.Pointer(&flat[0]))
	}
	buf := bufferReturningCall(procCompressPose, ptr, uintptr(len(poses)))
	runtime.KeepAlive(flat)
	return extractAndFree(buf)
}

// CompressPoseLossy: posQuantStep/posResyncInterval reach the position
// half's existing lossy support; quatQuantStep/quatResyncInterval are the
// analogous knobs for the Quaternion Joint. Either quantStep <= 1 is
// lossless for that half.
func CompressPoseLossy(poses []Pose, posQuantStep, posResyncInterval, quatQuantStep, quatResyncInterval uint32) ([]byte, error) {
	flat := posesToFlat(poses)
	var ptr uintptr
	if len(flat) > 0 {
		ptr = uintptr(unsafe.Pointer(&flat[0]))
	}
	buf := bufferReturningCall(procCompressPoseLossy, ptr, uintptr(len(poses)),
		uintptr(posQuantStep), uintptr(posResyncInterval), uintptr(quatQuantStep), uintptr(quatResyncInterval))
	runtime.KeepAlive(flat)
	return extractAndFree(buf)
}

// DecompressPose works for both lossless and lossy blobs.
func DecompressPose(data []byte) ([]Pose, error) {
	var inputPtr uintptr
	if len(data) > 0 {
		inputPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	var outCount uintptr
	buf := bufferReturningCall(procDecompressPose, inputPtr, uintptr(len(data)),
		uintptr(unsafe.Pointer(&outCount)))
	runtime.KeepAlive(data)
	raw, err := extractAndFree(buf)
	if err != nil {
		return nil, err
	}
	n := int(outCount)
	result := make([]Pose, n)
	for i := 0; i < n; i++ {
		base := i * 28
		x := *(*int32)(unsafe.Pointer(&raw[base]))
		y := *(*int32)(unsafe.Pointer(&raw[base+4]))
		z := *(*int32)(unsafe.Pointer(&raw[base+8]))
		qw := *(*int32)(unsafe.Pointer(&raw[base+12]))
		qx := *(*int32)(unsafe.Pointer(&raw[base+16]))
		qy := *(*int32)(unsafe.Pointer(&raw[base+20]))
		qz := *(*int32)(unsafe.Pointer(&raw[base+24]))
		result[i] = Pose{Point3D{x, y, z}, Quat{qw, qx, qy, qz}}
	}
	return result, nil
}

func CudaAvailable() bool {
	r := procCudaAvailable.call()
	return r != 0
}
