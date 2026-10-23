//! Authored queue ordering and host visibility through the native submission path.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_runtime::{
    Device, DeviceKind, DeviceOptions, EventRef, MessageSeverity, MessageSink, ResourceOptions,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn combined_submission_preserves_event_order_and_host_visible_writes() {
    let cache =
        Scratch(std::env::temp_dir().join(format!("m2v-queue-submit-{}", std::process::id())));
    std::fs::create_dir(&cache.0).unwrap();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let collected = Arc::clone(&errors);
    let mut options = DeviceOptions::default();
    options.kinds = if cfg!(m2v_cpu_icd) {
        vec![DeviceKind::Cpu]
    } else {
        vec![DeviceKind::Discrete, DeviceKind::Integrated]
    };
    options.layers = vec![c"VK_LAYER_KHRONOS_validation".to_owned()];
    options.shader_cache.directory = Some(cache.0.clone());
    options.messages = Some(MessageSink::new(move |message| {
        if message.kind.validation && message.severity == MessageSeverity::Error {
            collected.lock().unwrap().push(message.text.clone());
        }
    }));
    let device = Device::system_default_with(&options).unwrap();
    let first_queue = device.new_command_queue().unwrap();
    let second_queue = device.new_command_queue().unwrap();
    let bytes = device.new_buffer(256, ResourceOptions::shared()).unwrap();
    let event = device.new_shared_event().unwrap();
    let before = device.work_counts().combined_submissions;
    let first = first_queue.command_buffer().unwrap();
    let mut fill = first.blit_command_encoder().unwrap();
    fill.fill_buffer(&bytes, 0..256, 0x11).unwrap();
    fill.end_encoding().unwrap();
    first
        .encode_signal_event(EventRef::from(&event), 1)
        .unwrap();
    first.commit().unwrap();
    let second = second_queue.command_buffer().unwrap();
    second
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    let mut fill = second.blit_command_encoder().unwrap();
    fill.fill_buffer(&bytes, 0..128, 0x22).unwrap();
    fill.end_encoding().unwrap();
    second
        .encode_signal_event(EventRef::from(&event), 2)
        .unwrap();
    second.commit().unwrap();
    second
        .wait_until_completed_for(Duration::from_secs(10))
        .unwrap();
    first
        .wait_until_completed_for(Duration::from_secs(10))
        .unwrap();
    assert_eq!(first.error(), None);
    assert_eq!(second.error(), None);
    assert!(event.signaled_value() >= 2);
    // SAFETY: the shared buffer is live and both GPU writes have completed.
    let actual = unsafe { std::slice::from_raw_parts(bytes.contents().unwrap().as_ptr(), 256) };
    assert_eq!(&actual[..128], &[0x22; 128]);
    assert_eq!(&actual[128..], &[0x11; 128]);
    let empty = first_queue.command_buffer().unwrap();
    empty
        .encode_wait_for_event(EventRef::from(&event), 2)
        .unwrap();
    empty.commit().unwrap();
    empty
        .wait_until_completed_for(Duration::from_secs(10))
        .unwrap();
    assert_eq!(empty.error(), None);
    let combined = device.work_counts().combined_submissions - before;
    assert!(
        combined >= 1,
        "the combined submission path was not exercised"
    );
    eprintln!(
        "{}: {combined} combined submissions; ordered fills and host readback passed",
        device.name()
    );
    drop(empty);
    drop(second);
    drop(first);
    drop(event);
    drop(bytes);
    drop(second_queue);
    drop(first_queue);
    drop(device);
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "{errors:?}");
}
