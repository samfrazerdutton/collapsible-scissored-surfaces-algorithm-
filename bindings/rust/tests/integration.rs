// Smoke test for the Rust FFI bindings, run against the actual built
// shared library (not a mock) -- mirrors bindings/python/test_bindings.py
// so both bindings are held to the same real end-to-end bar.

fn lcg_next(seed: &mut u32) -> f64 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f64 / (1u32 << 24) as f64
}

#[test]
fn general_round_trip() {
    let data = "the quick brown fox jumps over the lazy dog. ".repeat(500);
    let data = data.as_bytes();
    let compressed = csa::compress(data, false).expect("compress failed");
    assert!(!compressed.is_empty());
    assert!(compressed.len() < data.len() / 10, "expected strong ratio on repetitive text");
    let back = csa::decompress(&compressed).expect("decompress failed");
    assert_eq!(back, data);
}

#[test]
fn geo2d_round_trip() {
    let mut points = Vec::new();
    let (mut x, mut y, mut dx, mut dy) = (50.0_f64, 0.0_f64, 3.0_f64, 0.0_f64);
    for _ in 0..300 {
        points.push((x.round() as i32, y.round() as i32));
        let ndx = 1.02 * (dx * 0.15_f64.cos() - dy * 0.15_f64.sin());
        let ndy = 1.02 * (dx * 0.15_f64.sin() + dy * 0.15_f64.cos());
        dx = ndx;
        dy = ndy;
        x += dx;
        y += dy;
    }

    let blob = csa::compress_geo2d(&points).expect("compress_geo2d failed");
    assert!(!blob.is_empty());
    let back = csa::decompress_geo2d(&blob).expect("decompress_geo2d failed");
    assert_eq!(back, points);
}

#[test]
fn geo2d_lossy_bounded_error() {
    let mut points = Vec::new();
    let (mut x, mut y, mut heading) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut seed = 7u32;
    for _ in 0..2000 {
        heading += (lcg_next(&mut seed) - 0.5) * 0.1;
        let speed = 8.0 + (lcg_next(&mut seed) - 0.5);
        x += speed * heading.cos();
        y += speed * heading.sin();
        points.push((x.round() as i32, y.round() as i32));
    }

    let lossless = csa::compress_geo2d(&points).unwrap();
    let lossy = csa::compress_geo2d_lossy(&points, 20, 64).unwrap();
    assert!(lossy.len() < lossless.len(), "lossy ({}) should beat lossless ({})", lossy.len(), lossless.len());

    let back = csa::decompress_geo2d(&lossy).unwrap();
    assert_eq!(back.len(), points.len());
    let max_err = points
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
        .max()
        .unwrap();
    assert!(max_err <= 20 * 64, "max_err={max_err} exceeds bound");
}

#[test]
fn geo3d_lossy_bounded_error() {
    let mut points = Vec::new();
    let (mut x, mut y, mut z, mut heading) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut seed = 314159u32;
    for _ in 0..2000 {
        heading += (lcg_next(&mut seed) - 0.5) * 0.08;
        let speed = 8.0 + (lcg_next(&mut seed) - 0.5);
        x += speed * heading.cos();
        y += speed * heading.sin();
        z += 3.0 + (lcg_next(&mut seed) - 0.5) * 0.5;
        points.push((x.round() as i32, y.round() as i32, z.round() as i32));
    }

    let lossless = csa::compress_geo3d(&points).unwrap();
    let back_lossless = csa::decompress_geo3d(&lossless).unwrap();
    assert_eq!(back_lossless, points);

    let lossy = csa::compress_geo3d_lossy(&points, 20, 64).unwrap();
    assert!(lossy.len() < lossless.len(), "3d lossy ({}) should beat lossless ({})", lossy.len(), lossless.len());

    let back = csa::decompress_geo3d(&lossy).unwrap();
    assert_eq!(back.len(), points.len());
    let max_err = points
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()).max((a.2 - b.2).abs()))
        .max()
        .unwrap();
    assert!(max_err <= 20 * 64, "max_err={max_err} exceeds bound");
}

#[test]
fn pose_lossy_bounded_error() {
    let mut poses = Vec::new();
    let (mut qw, mut qx, mut qy, mut qz) = (1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut seed = 4242u32;
    let mut z = 0.0_f64;
    let qscale = (1i64 << 20) as f64;
    for i in 0..1500 {
        let t = i as f64 * 0.05;
        let x = 2000.0 * t.cos();
        let y = 2000.0 * t.sin();
        z += 4.0 + (lcg_next(&mut seed) - 0.5) * 0.5;
        let position = (x.round() as i32, y.round() as i32, z.round() as i32);
        let orientation = (
            (qw * qscale).round() as i32,
            (qx * qscale).round() as i32,
            (qy * qscale).round() as i32,
            (qz * qscale).round() as i32,
        );
        poses.push((position, orientation));

        let deg = 2.0 + (lcg_next(&mut seed) - 0.5) * 0.3;
        let half = deg.to_radians() / 2.0;
        let (dqw, dqz) = (half.cos(), half.sin());
        let nw = qw * dqw - qz * dqz;
        let nx = qx * dqw + qy * dqz;
        let ny = qy * dqw - qx * dqz;
        let nz = qw * dqz + qz * dqw;
        qw = nw; qx = nx; qy = ny; qz = nz;
    }

    let lossless = csa::compress_pose(&poses).unwrap();
    let back_lossless = csa::decompress_pose(&lossless).unwrap();
    assert_eq!(back_lossless, poses);

    let lossy = csa::compress_pose_lossy(&poses, 8, 64, 32, 32).unwrap();
    assert!(lossy.len() < lossless.len(), "pose lossy ({}) should beat lossless ({})", lossy.len(), lossless.len());

    let back = csa::decompress_pose(&lossy).unwrap();
    assert_eq!(back.len(), poses.len());
    let max_err = poses
        .iter()
        .zip(back.iter())
        .map(|(a, b)| {
            let pos_err = (a.0 .0 - b.0 .0).abs().max((a.0 .1 - b.0 .1).abs()).max((a.0 .2 - b.0 .2).abs());
            let quat_err = (a.1 .0 - b.1 .0).abs().max((a.1 .1 - b.1 .1).abs()).max((a.1 .2 - b.1 .2).abs()).max((a.1 .3 - b.1 .3).abs());
            pos_err.max(quat_err)
        })
        .max()
        .unwrap();
    assert!(max_err <= 64 * 64, "max_err={max_err} exceeds bound");
}

#[test]
fn error_handling_on_garbage_input() {
    let garbage = [1u8, 2, 3, 4, 5];
    let err = csa::decompress(&garbage).expect_err("expected an error on garbage input");
    assert!(!err.0.is_empty());
}

#[test]
fn cuda_available_does_not_panic() {
    let _ = csa::cuda_available();
}
