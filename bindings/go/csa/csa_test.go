// Smoke test for the Go bindings, run against the actual built shared
// library (not a mock) -- mirrors bindings/python/test_bindings.py,
// bindings/rust/tests/integration.rs, and bindings/csharp/Csa.Tests so
// all four bindings are held to the same real end-to-end bar.
package csa

import (
	"math"
	"strings"
	"testing"
)

type lcg struct{ seed uint32 }

func (l *lcg) next() float64 {
	l.seed = l.seed*1664525 + 1013904223
	return float64(l.seed>>8) / float64(uint32(1)<<24)
}

func TestGeneralRoundTrip(t *testing.T) {
	data := []byte(strings.Repeat("the quick brown fox jumps over the lazy dog. ", 500))
	compressed, err := Compress(data, false)
	if err != nil {
		t.Fatalf("compress failed: %v", err)
	}
	if len(compressed) == 0 || len(compressed) >= len(data)/10 {
		t.Fatalf("expected strong ratio on repetitive text, got %d -> %d", len(data), len(compressed))
	}
	back, err := Decompress(compressed)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	if string(back) != string(data) {
		t.Fatal("round-trip mismatch")
	}
}

func TestGeo2DRoundTrip(t *testing.T) {
	var points []Point2D
	x, y, dx, dy := 50.0, 0.0, 3.0, 0.0
	for i := 0; i < 300; i++ {
		points = append(points, Point2D{int32(math.Round(x)), int32(math.Round(y))})
		ndx := 1.02 * (dx*math.Cos(0.15) - dy*math.Sin(0.15))
		ndy := 1.02 * (dx*math.Sin(0.15) + dy*math.Cos(0.15))
		dx, dy = ndx, ndy
		x += dx
		y += dy
	}

	blob, err := CompressGeo2D(points)
	if err != nil {
		t.Fatalf("compress_geo2d failed: %v", err)
	}
	if len(blob) == 0 {
		t.Fatal("expected nonempty blob")
	}
	back, err := DecompressGeo2D(blob)
	if err != nil {
		t.Fatalf("decompress_geo2d failed: %v", err)
	}
	if len(back) != len(points) {
		t.Fatalf("length mismatch: %d vs %d", len(back), len(points))
	}
	for i := range points {
		if back[i] != points[i] {
			t.Fatalf("point %d mismatch: got %v want %v", i, back[i], points[i])
		}
	}
}

func TestGeo2DLossyBoundedError(t *testing.T) {
	var points []Point2D
	x, y, heading := 0.0, 0.0, 0.0
	g := &lcg{seed: 7}
	for i := 0; i < 2000; i++ {
		heading += (g.next() - 0.5) * 0.1
		speed := 8.0 + (g.next() - 0.5)
		x += speed * math.Cos(heading)
		y += speed * math.Sin(heading)
		points = append(points, Point2D{int32(math.Round(x)), int32(math.Round(y))})
	}

	lossless, err := CompressGeo2D(points)
	if err != nil {
		t.Fatalf("lossless compress failed: %v", err)
	}
	lossy, err := CompressGeo2DLossy(points, 20, 64)
	if err != nil {
		t.Fatalf("lossy compress failed: %v", err)
	}
	if len(lossy) >= len(lossless) {
		t.Fatalf("expected lossy (%d) smaller than lossless (%d)", len(lossy), len(lossless))
	}

	back, err := DecompressGeo2D(lossy)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	if len(back) != len(points) {
		t.Fatalf("length mismatch: %d vs %d", len(back), len(points))
	}
	maxErr := int32(0)
	for i := range points {
		dx := points[i].X - back[i].X
		if dx < 0 {
			dx = -dx
		}
		dy := points[i].Y - back[i].Y
		if dy < 0 {
			dy = -dy
		}
		if dx > maxErr {
			maxErr = dx
		}
		if dy > maxErr {
			maxErr = dy
		}
	}
	if maxErr > 20*64 {
		t.Fatalf("max_err=%d exceeds bound", maxErr)
	}
	t.Logf("2d lossy: %d -> %d bytes, max coordinate error = %d", len(lossless), len(lossy), maxErr)
}

func TestGeo3DLossyBoundedError(t *testing.T) {
	var points []Point3D
	x, y, z, heading := 0.0, 0.0, 0.0, 0.0
	g := &lcg{seed: 314159}
	for i := 0; i < 2000; i++ {
		heading += (g.next() - 0.5) * 0.08
		speed := 8.0 + (g.next() - 0.5)
		x += speed * math.Cos(heading)
		y += speed * math.Sin(heading)
		z += 3.0 + (g.next()-0.5)*0.5
		points = append(points, Point3D{int32(math.Round(x)), int32(math.Round(y)), int32(math.Round(z))})
	}

	lossless, err := CompressGeo3D(points)
	if err != nil {
		t.Fatalf("lossless compress failed: %v", err)
	}
	backLossless, err := DecompressGeo3D(lossless)
	if err != nil {
		t.Fatalf("lossless decompress failed: %v", err)
	}
	for i := range points {
		if backLossless[i] != points[i] {
			t.Fatalf("lossless point %d mismatch: got %v want %v", i, backLossless[i], points[i])
		}
	}

	lossy, err := CompressGeo3DLossy(points, 20, 64)
	if err != nil {
		t.Fatalf("lossy compress failed: %v", err)
	}
	if len(lossy) >= len(lossless) {
		t.Fatalf("expected 3d lossy (%d) smaller than lossless (%d)", len(lossy), len(lossless))
	}

	back, err := DecompressGeo3D(lossy)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	maxErr := int32(0)
	abs := func(v int32) int32 {
		if v < 0 {
			return -v
		}
		return v
	}
	for i := range points {
		if d := abs(points[i].X - back[i].X); d > maxErr {
			maxErr = d
		}
		if d := abs(points[i].Y - back[i].Y); d > maxErr {
			maxErr = d
		}
		if d := abs(points[i].Z - back[i].Z); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 20*64 {
		t.Fatalf("max_err=%d exceeds bound", maxErr)
	}
	t.Logf("3d lossy: %d -> %d bytes, max coordinate error = %d", len(lossless), len(lossy), maxErr)
}

func TestPoseLossyBoundedError(t *testing.T) {
	var poses []Pose
	qw, qx, qy, qz := 1.0, 0.0, 0.0, 0.0
	g := &lcg{seed: 4242}
	z := 0.0
	const qscale = float64(1 << 20)
	for i := 0; i < 1500; i++ {
		t := float64(i) * 0.05
		x, y := 2000*math.Cos(t), 2000*math.Sin(t)
		z += 4.0 + (g.next()-0.5)*0.5
		position := Point3D{int32(math.Round(x)), int32(math.Round(y)), int32(math.Round(z))}
		orientation := Quat{
			int32(math.Round(qw * qscale)), int32(math.Round(qx * qscale)),
			int32(math.Round(qy * qscale)), int32(math.Round(qz * qscale)),
		}
		poses = append(poses, Pose{position, orientation})

		deg := 2.0 + (g.next()-0.5)*0.3
		half := deg * math.Pi / 180.0 / 2.0
		dqw, dqz := math.Cos(half), math.Sin(half)
		nw := qw*dqw - qz*dqz
		nx := qx*dqw + qy*dqz
		ny := qy*dqw - qx*dqz
		nz := qw*dqz + qz*dqw
		qw, qx, qy, qz = nw, nx, ny, nz
	}

	lossless, err := CompressPose(poses)
	if err != nil {
		t.Fatalf("lossless compress failed: %v", err)
	}
	backLossless, err := DecompressPose(lossless)
	if err != nil {
		t.Fatalf("lossless decompress failed: %v", err)
	}
	for i := range poses {
		if backLossless[i] != poses[i] {
			t.Fatalf("lossless pose %d mismatch: got %v want %v", i, backLossless[i], poses[i])
		}
	}

	lossy, err := CompressPoseLossy(poses, 8, 64, 32, 32)
	if err != nil {
		t.Fatalf("lossy compress failed: %v", err)
	}
	if len(lossy) >= len(lossless) {
		t.Fatalf("expected pose lossy (%d) smaller than lossless (%d)", len(lossy), len(lossless))
	}

	back, err := DecompressPose(lossy)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	maxErr := int32(0)
	abs := func(v int32) int32 {
		if v < 0 {
			return -v
		}
		return v
	}
	for i := range poses {
		a, b := poses[i], back[i]
		for _, d := range []int32{
			abs(a.Position.X - b.Position.X), abs(a.Position.Y - b.Position.Y), abs(a.Position.Z - b.Position.Z),
			abs(a.Orientation.W - b.Orientation.W), abs(a.Orientation.X - b.Orientation.X),
			abs(a.Orientation.Y - b.Orientation.Y), abs(a.Orientation.Z - b.Orientation.Z),
		} {
			if d > maxErr {
				maxErr = d
			}
		}
	}
	if maxErr > 64*64 {
		t.Fatalf("max_err=%d exceeds bound", maxErr)
	}
	t.Logf("pose lossy: %d -> %d bytes, max component error = %d", len(lossless), len(lossy), maxErr)
}

func TestErrorHandlingOnGarbageInput(t *testing.T) {
	_, err := Decompress([]byte{1, 2, 3, 4, 5})
	if err == nil {
		t.Fatal("expected an error on garbage input")
	}
	if err.Error() == "" {
		t.Fatal("expected a non-empty error message")
	}
}

func TestCudaAvailableDoesNotPanic(t *testing.T) {
	_ = CudaAvailable()
}
