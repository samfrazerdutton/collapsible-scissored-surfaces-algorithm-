// P/Invoke declarations for libcsa's stable C ABI (include/csa/csa_capi.h).
// Kept separate from the safe public wrapper (Csa.cs) so the unsafe
// surface is easy to audit in one place.
using System;
using System.Runtime.InteropServices;

namespace Csa
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct CsaBuffer
    {
        public IntPtr Data;
        public UIntPtr Size;
    }

    internal static class CsaNative
    {
        private const string LibName = "csa";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_compress(byte[] input, UIntPtr input_size, int use_gpu);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_decompress(byte[] input, UIntPtr input_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_compress_geo2d(int[] xy, UIntPtr count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_compress_geo2d_lossy(int[] xy, UIntPtr count, uint quant_step, uint resync_interval);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_decompress_geo2d(byte[] input, UIntPtr input_size, out UIntPtr out_count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_compress_geo3d(int[] xyz, UIntPtr count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_compress_geo3d_lossy(int[] xyz, UIntPtr count, uint quant_step, uint resync_interval);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_decompress_geo3d(byte[] input, UIntPtr input_size, out UIntPtr out_count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_compress_pose(int[] pose7, UIntPtr count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_compress_pose_lossy(int[] pose7, UIntPtr count,
            uint pos_quant_step, uint pos_resync_interval, uint quat_quant_step, uint quat_resync_interval);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CsaBuffer csa_decompress_pose(byte[] input, UIntPtr input_size, out UIntPtr out_count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void csa_free_buffer(CsaBuffer buf);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr csa_last_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int csa_cuda_available();
    }
}
