//! Native Windows texture copies and lifetimes on a discrete GPU, without guest shaders.
#![cfg(all(windows, m2v_gpu))]

use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{
    BlitOption, CpuCacheMode, Device, DeviceKind, DeviceOptions, HazardTrackingMode,
    MessageSeverity, MessageSink, Origin, Region, ResourceOptions, Size, StorageMode,
    TextureDescriptor,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[test]
fn cpu_gpu_texture_copies_and_small_compressed_levels_agree() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let collected = Arc::clone(&errors);
    let mut options = DeviceOptions::default();
    options.kinds = vec![DeviceKind::Discrete];
    options.layers = vec![c"VK_LAYER_KHRONOS_validation".to_owned()];
    options.messages = Some(MessageSink::new(move |message| {
        if message.kind.validation && message.severity == MessageSeverity::Error {
            collected.lock().unwrap().push(message.text.clone());
        }
    }));
    let device = Device::system_default_with(&options).unwrap();
    let queue = device.new_command_queue().unwrap();
    let initial = device.work_counts();
    for mode in [StorageMode::Shared, StorageMode::Managed] {
        for side in [16u64, 256, 1024] {
            let length = (side * side * 4) as usize;
            let region = Region::new_2d(0, 0, side, side);
            for seed in 0..4 {
                let mut descriptor =
                    TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
                descriptor.resource_options = ResourceOptions::new(
                    mode,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Default,
                );
                let texture = device.new_texture(&descriptor).unwrap();
                let bytes: Vec<u8> = (0..length)
                    .map(|i| ((i * 29 + seed * 7) % 251) as u8)
                    .collect();
                texture
                    .replace_region(region, 0, 0, &bytes, side * 4, 0)
                    .unwrap();
                let readback = device
                    .new_buffer(length as u64, ResourceOptions::shared())
                    .unwrap();
                let command = queue.command_buffer().unwrap();
                let mut blit = command.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(side, side, 1),
                    &readback,
                    0,
                    side * 4,
                    length as u64,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command.commit().unwrap();
                command
                    .wait_until_completed_for(Duration::from_secs(10))
                    .unwrap();
                // SAFETY: the buffer remains live and the GPU copy has completed.
                let actual = unsafe {
                    std::slice::from_raw_parts(readback.contents().unwrap().as_ptr(), length)
                };
                assert_eq!(actual, bytes, "CPU -> GPU, {side}, {mode:?}, {seed}");
                let written: Vec<u8> = bytes.iter().map(|b| b ^ 0xa5).collect();
                // SAFETY: the prior GPU work completed; only this thread writes these bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        written.as_ptr(),
                        readback.contents().unwrap().as_ptr(),
                        length,
                    );
                }
                let command = queue.command_buffer().unwrap();
                let mut blit = command.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_texture(
                    &readback,
                    0,
                    side * 4,
                    length as u64,
                    Size::new(side, side, 1),
                    &texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command.commit().unwrap();
                command
                    .wait_until_completed_for(Duration::from_secs(10))
                    .unwrap();
                let mut actual = vec![0; length];
                texture
                    .get_bytes(&mut actual, side * 4, 0, region, 0, 0)
                    .unwrap();
                assert_eq!(actual, written, "GPU -> CPU, {side}, {mode:?}, {seed}");
            }
        }
    }
    let compressed = |w, h| {
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BC3RGBA, w, h, false);
        descriptor.resource_options = ResourceOptions::shared();
        device.new_texture(&descriptor).unwrap()
    };
    let small = compressed(4, 2);
    let big = compressed(8, 8);
    let block = [0x39u8; 16];
    small
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &block, 16, 0)
        .unwrap();
    let command = queue.command_buffer().unwrap();
    let mut blit = command.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_texture(
        &small,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(4, 4, 1),
        &big,
        0,
        0,
        Origin::new(0, 0, 0),
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command.commit().unwrap();
    command
        .wait_until_completed_for(Duration::from_secs(10))
        .unwrap();
    let mut actual = [0u8; 64];
    big.get_bytes(&mut actual, 32, 0, Region::new_2d(0, 0, 8, 8), 0, 0)
        .unwrap();
    assert_eq!(&actual[..16], &block);
    assert!(actual[16..].iter().all(|&b| b == 0));
    eprintln!(
        "{}: 24 CPU/GPU texture rounds; host-created images={}; BC3 edge block passed",
        device.name(),
        device.work_counts().images_started_on_the_host - initial.images_started_on_the_host
    );
    drop(small);
    drop(big);
    drop(queue);
    drop(device);
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "{errors:?}");
}
