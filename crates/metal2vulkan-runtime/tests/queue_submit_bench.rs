//! Native throughput of small, ordered command buffers; no guest shaders.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_runtime::{Device, DeviceKind, DeviceOptions, ResourceOptions};
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "native GPU throughput comparison; run explicitly with --ignored --nocapture"]
fn small_ordered_submissions_throughput() {
    let cache =
        Scratch(std::env::temp_dir().join(format!("m2v-submit-throughput-{}", std::process::id())));
    std::fs::create_dir(&cache.0).unwrap();
    let mut options = DeviceOptions::default();
    options.kinds = if cfg!(m2v_cpu_icd) {
        vec![DeviceKind::Cpu]
    } else {
        vec![DeviceKind::Discrete, DeviceKind::Integrated]
    };
    options.shader_cache.directory = Some(cache.0.clone());
    let device = Device::system_default_with(&options).unwrap();
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let mut samples = Vec::new();
    for round in 0..10 {
        let mut commands = Vec::with_capacity(128);
        let start = Instant::now();
        for value in 0..128 {
            let command = queue.command_buffer().unwrap();
            let mut blit = command.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..4096, value).unwrap();
            blit.end_encoding().unwrap();
            command.commit().unwrap();
            commands.push(command);
        }
        for command in &commands {
            command
                .wait_until_completed_for(Duration::from_secs(10))
                .unwrap();
            assert_eq!(command.error(), None);
        }
        let micros = start.elapsed().as_secs_f64() * 1e6 / 128.0;
        // SAFETY: the shared buffer is live and every GPU write completed.
        let actual =
            unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), 4096) };
        assert!(actual.iter().all(|&byte| byte == 127));
        if round > 0 {
            samples.push(micros);
        }
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "SUBMIT_BENCH device={:?} us_per_cb median={:.3} min={:.3} max={:.3} rounds=9 count=128",
        device.name(),
        samples[4],
        samples[0],
        samples[8]
    );
    drop(buffer);
    drop(queue);
    drop(device);
}
