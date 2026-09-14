// Smoke test for the C# bindings, run against the actual built shared
// library (not a mock) -- mirrors bindings/python/test_bindings.py and
// bindings/rust/tests/integration.rs so all three bindings are held to
// the same real end-to-end bar.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Csa;

int checks = 0;
int failures = 0;

void Check(bool cond, string label)
{
    checks++;
    if (!cond)
    {
        failures++;
        Console.WriteLine($"FAIL: {label}");
    }
}

uint lcgSeed = 0;
double LcgNext()
{
    lcgSeed = unchecked(lcgSeed * 1664525u + 1013904223u);
    return (double)(lcgSeed >> 8) / (double)(1u << 24);
}

void TestGeneral()
{
    var sb = new StringBuilder();
    for (int i = 0; i < 500; i++) sb.Append("the quick brown fox jumps over the lazy dog. ");
    byte[] data = Encoding.ASCII.GetBytes(sb.ToString());

    byte[] compressed = Codec.Compress(data);
    Check(compressed.Length > 0 && compressed.Length < data.Length / 10, "general compress ratio");
    byte[] back = Codec.Decompress(compressed);
    Check(back.SequenceEqual(data), "general round-trip");
}

void TestGeo2D()
{
    var points = new List<(int X, int Y)>();
    double x = 50, y = 0, dx = 3, dy = 0;
    for (int i = 0; i < 300; i++)
    {
        points.Add(((int)Math.Round(x), (int)Math.Round(y)));
        double ndx = 1.02 * (dx * Math.Cos(0.15) - dy * Math.Sin(0.15));
        double ndy = 1.02 * (dx * Math.Sin(0.15) + dy * Math.Cos(0.15));
        dx = ndx; dy = ndy;
        x += dx; y += dy;
    }

    byte[] blob = Codec.CompressGeo2D(points);
    Check(blob.Length > 0, "geo2d compress produced output");
    var back = Codec.DecompressGeo2D(blob);
    Check(back.SequenceEqual(points), "geo2d exact round-trip");
}

void TestGeo2DLossy()
{
    var points = new List<(int X, int Y)>();
    double x = 0, y = 0, heading = 0;
    lcgSeed = 7;
    for (int i = 0; i < 2000; i++)
    {
        heading += (LcgNext() - 0.5) * 0.1;
        double speed = 8.0 + (LcgNext() - 0.5);
        x += speed * Math.Cos(heading);
        y += speed * Math.Sin(heading);
        points.Add(((int)Math.Round(x), (int)Math.Round(y)));
    }

    byte[] lossless = Codec.CompressGeo2D(points);
    byte[] lossy = Codec.CompressGeo2DLossy(points, 20, 64);
    Check(lossy.Length < lossless.Length, $"lossy smaller than lossless ({lossy.Length} < {lossless.Length})");

    var back = Codec.DecompressGeo2D(lossy);
    Check(back.Length == points.Count, "lossy round-trip length matches");
    int maxErr = 0;
    for (int i = 0; i < points.Count; i++)
    {
        maxErr = Math.Max(maxErr, Math.Max(Math.Abs(points[i].X - back[i].X), Math.Abs(points[i].Y - back[i].Y)));
    }
    Check(maxErr <= 20 * 64, $"lossy error bounded (maxErr={maxErr})");
    Console.WriteLine($"  lossy: {lossless.Length} -> {lossy.Length} bytes, max coordinate error = {maxErr}");
}

void TestGeo3DLossy()
{
    var points = new List<(int X, int Y, int Z)>();
    double x = 0, y = 0, z = 0, heading = 0;
    lcgSeed = 314159;
    for (int i = 0; i < 2000; i++)
    {
        heading += (LcgNext() - 0.5) * 0.08;
        double speed = 8.0 + (LcgNext() - 0.5);
        x += speed * Math.Cos(heading);
        y += speed * Math.Sin(heading);
        z += 3.0 + (LcgNext() - 0.5) * 0.5;
        points.Add(((int)Math.Round(x), (int)Math.Round(y), (int)Math.Round(z)));
    }

    byte[] lossless = Codec.CompressGeo3D(points);
    var backLossless = Codec.DecompressGeo3D(lossless);
    Check(backLossless.SequenceEqual(points), "geo3d exact round-trip");

    byte[] lossy = Codec.CompressGeo3DLossy(points, 20, 64);
    Check(lossy.Length < lossless.Length, $"3d lossy smaller than lossless ({lossy.Length} < {lossless.Length})");

    var back = Codec.DecompressGeo3D(lossy);
    Check(back.Length == points.Count, "3d lossy round-trip length matches");
    int maxErr = 0;
    for (int i = 0; i < points.Count; i++)
    {
        maxErr = Math.Max(maxErr, Math.Max(Math.Abs(points[i].X - back[i].X),
                 Math.Max(Math.Abs(points[i].Y - back[i].Y), Math.Abs(points[i].Z - back[i].Z))));
    }
    Check(maxErr <= 20 * 64, $"3d lossy error bounded (maxErr={maxErr})");
    Console.WriteLine($"  3d lossy: {lossless.Length} -> {lossy.Length} bytes, max coordinate error = {maxErr}");
}

void TestPoseLossy()
{
    var poses = new List<Codec.Pose>();
    double qw = 1, qx = 0, qy = 0, qz = 0;
    lcgSeed = 4242;
    double z = 0;
    double qscale = 1 << 20;
    for (int i = 0; i < 1500; i++)
    {
        double t = i * 0.05;
        double x = 2000 * Math.Cos(t), y = 2000 * Math.Sin(t);
        z += 4.0 + (LcgNext() - 0.5) * 0.5;
        var position = ((int)Math.Round(x), (int)Math.Round(y), (int)Math.Round(z));
        var orientation = ((int)Math.Round(qw * qscale), (int)Math.Round(qx * qscale),
                            (int)Math.Round(qy * qscale), (int)Math.Round(qz * qscale));
        poses.Add(new Codec.Pose(position, orientation));

        double deg = 2.0 + (LcgNext() - 0.5) * 0.3;
        double half = deg * Math.PI / 180.0 / 2.0;
        double dqw = Math.Cos(half), dqz = Math.Sin(half);
        double nw = qw * dqw - qz * dqz;
        double nx = qx * dqw + qy * dqz;
        double ny = qy * dqw - qx * dqz;
        double nz = qw * dqz + qz * dqw;
        qw = nw; qx = nx; qy = ny; qz = nz;
    }

    byte[] lossless = Codec.CompressPose(poses);
    var backLossless = Codec.DecompressPose(lossless);
    Check(backLossless.SequenceEqual(poses), "pose exact round-trip");

    byte[] lossy = Codec.CompressPoseLossy(poses, 8, 64, 32, 32);
    Check(lossy.Length < lossless.Length, $"pose lossy smaller than lossless ({lossy.Length} < {lossless.Length})");

    var back = Codec.DecompressPose(lossy);
    Check(back.Length == poses.Count, "pose lossy round-trip length matches");
    int maxErr = 0;
    for (int i = 0; i < poses.Count; i++)
    {
        var a = poses[i]; var b = back[i];
        maxErr = Math.Max(maxErr, Math.Max(Math.Abs(a.Position.X - b.Position.X),
                 Math.Max(Math.Abs(a.Position.Y - b.Position.Y), Math.Abs(a.Position.Z - b.Position.Z))));
        maxErr = Math.Max(maxErr, Math.Max(Math.Abs(a.Orientation.W - b.Orientation.W),
                 Math.Max(Math.Abs(a.Orientation.X - b.Orientation.X),
                 Math.Max(Math.Abs(a.Orientation.Y - b.Orientation.Y), Math.Abs(a.Orientation.Z - b.Orientation.Z)))));
    }
    Check(maxErr <= 64 * 64, $"pose lossy error bounded (maxErr={maxErr})");
    Console.WriteLine($"  pose lossy: {lossless.Length} -> {lossy.Length} bytes, max component error = {maxErr}");
}

void TestErrorHandling()
{
    try
    {
        Codec.Decompress(new byte[] { 1, 2, 3, 4, 5 });
        Check(false, "expected CsaException on garbage input");
    }
    catch (CsaException e)
    {
        Check(e.Message.Length > 0, "CsaException has a message");
    }
}

TestGeneral();
TestGeo2D();
TestGeo2DLossy();
TestGeo3DLossy();
TestPoseLossy();
TestErrorHandling();
Console.WriteLine($"cuda_available() = {Codec.CudaAvailable()}");
Console.WriteLine($"{checks} checks, {failures} failures");
return failures == 0 ? 0 : 1;
