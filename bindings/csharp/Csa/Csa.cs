// Public C# API for libcsa, on top of the P/Invoke declarations in
// CsaNative.cs -- mirrors bindings/python/csa.py and bindings/rust/src/lib.rs
// so all three bindings agree on behavior (all three are built on the same
// C ABI in include/csa/csa_capi.h).
using System;
using System.Collections.Generic;

namespace Csa
{
    /// <summary>An error reported by libcsa, sourced from csa_last_error().</summary>
    public class CsaException : Exception
    {
        public CsaException(string message) : base(message) { }
    }

    public static class Codec
    {
        // A genuinely empty result (e.g. decompressing an empty file) looks
        // identical to a failed call at the struct level (Data == IntPtr.Zero,
        // Size == 0) -- csa_last_error() disambiguates, and it's cleared on
        // every success, same reasoning as the Python/Rust bindings.
        private static byte[] CheckAndExtract(CsaBuffer buf)
        {
            if (buf.Data == IntPtr.Zero && buf.Size == UIntPtr.Zero)
            {
                IntPtr errPtr = CsaNative.csa_last_error();
                if (errPtr != IntPtr.Zero)
                {
                    string msg = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(errPtr) ?? "";
                    if (msg.Length > 0) throw new CsaException(msg);
                }
                return Array.Empty<byte>();
            }
            int size = checked((int)buf.Size);
            byte[] result = new byte[size];
            System.Runtime.InteropServices.Marshal.Copy(buf.Data, result, 0, size);
            CsaNative.csa_free_buffer(buf);
            return result;
        }

        /// <summary>
        /// General-purpose compression: tries raw storage, the Pantograph
        /// Lift, and the LZ dictionary matcher, and keeps whichever encodes
        /// smallest. useGpu tries the CUDA path for the Pantograph Lift
        /// candidate if a device is available, falling back transparently
        /// otherwise.
        /// </summary>
        public static byte[] Compress(byte[] data, bool useGpu = false)
        {
            var buf = CsaNative.csa_compress(data, (UIntPtr)data.Length, useGpu ? 1 : 0);
            return CheckAndExtract(buf);
        }

        public static byte[] Decompress(byte[] data)
        {
            var buf = CsaNative.csa_decompress(data, (UIntPtr)data.Length);
            return CheckAndExtract(buf);
        }

        private static int[] Points2DToFlat(IReadOnlyList<(int X, int Y)> points)
        {
            var flat = new int[points.Count * 2];
            for (int i = 0; i < points.Count; i++)
            {
                flat[2 * i] = points[i].X;
                flat[2 * i + 1] = points[i].Y;
            }
            return flat;
        }

        private static int[] Points3DToFlat(IReadOnlyList<(int X, int Y, int Z)> points)
        {
            var flat = new int[points.Count * 3];
            for (int i = 0; i < points.Count; i++)
            {
                flat[3 * i] = points[i].X;
                flat[3 * i + 1] = points[i].Y;
                flat[3 * i + 2] = points[i].Z;
            }
            return flat;
        }

        public static byte[] CompressGeo2D(IReadOnlyList<(int X, int Y)> points)
        {
            var flat = Points2DToFlat(points);
            var buf = CsaNative.csa_compress_geo2d(flat, (UIntPtr)points.Count);
            return CheckAndExtract(buf);
        }

        /// <summary>
        /// quantStep &lt;= 1 is lossless (identical to CompressGeo2D).
        /// resyncInterval periodically stores an exact rod to bound how far
        /// absolute position error can drift (0 = never); see DESIGN.md.
        /// </summary>
        public static byte[] CompressGeo2DLossy(IReadOnlyList<(int X, int Y)> points, uint quantStep, uint resyncInterval = 0)
        {
            var flat = Points2DToFlat(points);
            var buf = CsaNative.csa_compress_geo2d_lossy(flat, (UIntPtr)points.Count, quantStep, resyncInterval);
            return CheckAndExtract(buf);
        }

        /// <summary>
        /// Works for both lossless and lossy blobs -- the quantization
        /// parameters ride in the blob itself, there is no separate lossy
        /// decode entry point.
        /// </summary>
        public static (int X, int Y)[] DecompressGeo2D(byte[] data)
        {
            var buf = CsaNative.csa_decompress_geo2d(data, (UIntPtr)data.Length, out UIntPtr outCount);
            byte[] raw = CheckAndExtract(buf);
            int n = checked((int)outCount);
            var result = new (int X, int Y)[n];
            for (int i = 0; i < n; i++)
            {
                int x = BitConverter.ToInt32(raw, i * 8);
                int y = BitConverter.ToInt32(raw, i * 8 + 4);
                result[i] = (x, y);
            }
            return result;
        }

        public static byte[] CompressGeo3D(IReadOnlyList<(int X, int Y, int Z)> points)
        {
            var flat = Points3DToFlat(points);
            var buf = CsaNative.csa_compress_geo3d(flat, (UIntPtr)points.Count);
            return CheckAndExtract(buf);
        }

        /// <summary>
        /// Same design as CompressGeo2DLossy, on the true 3D similarity
        /// joint. quantStep &lt;= 1 is lossless (identical to CompressGeo3D).
        /// </summary>
        public static byte[] CompressGeo3DLossy(IReadOnlyList<(int X, int Y, int Z)> points, uint quantStep, uint resyncInterval = 0)
        {
            var flat = Points3DToFlat(points);
            var buf = CsaNative.csa_compress_geo3d_lossy(flat, (UIntPtr)points.Count, quantStep, resyncInterval);
            return CheckAndExtract(buf);
        }

        public static (int X, int Y, int Z)[] DecompressGeo3D(byte[] data)
        {
            var buf = CsaNative.csa_decompress_geo3d(data, (UIntPtr)data.Length, out UIntPtr outCount);
            byte[] raw = CheckAndExtract(buf);
            int n = checked((int)outCount);
            var result = new (int X, int Y, int Z)[n];
            for (int i = 0; i < n; i++)
            {
                int x = BitConverter.ToInt32(raw, i * 12);
                int y = BitConverter.ToInt32(raw, i * 12 + 4);
                int z = BitConverter.ToInt32(raw, i * 12 + 8);
                result[i] = (x, y, z);
            }
            return result;
        }

        public static bool CudaAvailable() => CsaNative.csa_cuda_available() != 0;
    }
}
