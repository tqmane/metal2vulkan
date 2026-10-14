//! A conservative Windows Function-storage bound, using validation's authored private-array kernel.
#![cfg(all(windows, m2v_gpu))]

use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ComputePipelineState, Device, DeviceKind, DeviceOptions, MessageSeverity, MessageSink,
    PipelineOptions, ResourceOptions, Size,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn read_buffer(
    buffer: &metal2vulkan_runtime::Buffer,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>, String> {
    let at = buffer.contents().ok_or("unmapped readback")?;
    assert!(offset + length <= buffer.length());
    // SAFETY: the caller completed GPU writes and owns the buffer throughout this copy.
    Ok(
        unsafe { std::slice::from_raw_parts(at.as_ptr().add(offset as usize), length as usize) }
            .to_vec(),
    )
}

/// The longest wait for one run's command buffer (the EVs bound a whole size at 60 s).
const WAIT: Duration = Duration::from_secs(10);

/// The read order's stride: prime, so steps `0..lines` visit every line of the sizes measured.
const STRIDE: u32 = 7919;

/// The words of a 64-byte line.
const LINE: u32 = 16;

/// The kernel of a private array of `words` words (a multiple of [`LINE`]).
fn kernel(words: u32) -> String {
    let lines = words / LINE;
    format!(
        r#"; Owned synthetic kernel (WP-UNI): a private array of {words} words a thread.
source_filename = "private_memory.metal"

define void @private_words(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [{words} x i32], align 4
  br label %fill
fill:
  %i = phi i32 [ 0, %entry ], [ %i1, %fill ]
  %w = mul i32 %i, {LINE}
  %p = getelementptr inbounds [{words} x i32], ptr %a, i32 0, i32 %w
  %v = xor i32 %i, %gid
  store i32 %v, ptr %p, align 4
  %i1 = add nuw i32 %i, 1
  %filled = icmp eq i32 %i1, {lines}
  br i1 %filled, label %read, label %fill
read:
  %j = phi i32 [ 0, %fill ], [ %j1, %read ]
  %s = phi i32 [ 0, %fill ], [ %s1, %read ]
  %m = mul i32 %j, {STRIDE}
  %k = add i32 %m, %gid
  %line = urem i32 %k, {lines}
  %idx = mul i32 %line, {LINE}
  %q = getelementptr inbounds [{words} x i32], ptr %a, i32 0, i32 %idx
  %x = load i32, ptr %q, align 4
  %j1 = add nuw i32 %j, 1
  %y = mul i32 %x, %j1
  %s1 = add i32 %s, %y
  %done = icmp eq i32 %j1, {lines}
  br i1 %done, label %write, label %read
write:
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %s1, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @private_words, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

/// What thread `gid`'s output word holds.
fn expected(words: u32, gid: u32) -> u32 {
    let lines = words / LINE;
    (0..lines).fold(0u32, |s, j| {
        let line = (j.wrapping_mul(STRIDE).wrapping_add(gid)) % lines;
        s.wrapping_add((line ^ gid).wrapping_mul(j + 1))
    })
}

/// The pipeline state of the kernel of `words` words on `device`, or what refused it.
fn state(device: &Device, words: u32) -> Result<ComputePipelineState, String> {
    state_of(device, &kernel(words), "private_words")
}

/// The pipeline state of kernel `entry` of the module `text` on `device`, or what refused it.
fn state_of(device: &Device, text: &str, entry: &str) -> Result<ComputePipelineState, String> {
    let parsed = metal2vulkan::library::Library::parse(&llvm_assemble_text(text).unwrap())
        .map_err(|e| format!("library refused: {e}"))?;
    let library = device.new_library_with_compile_library(parsed);
    let function = library.new_function_with_name(entry).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .map_err(|e| format!("state refused: {e}"))
}

/// Run `state` over `threads` threads: whether every checked thread's output word is what
/// `expected` says of its id, and what the run answered. The first run of a state also creates
/// its `VkPipeline`, so a refused creation is a refused dispatch or commit.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    threads: u32,
    expected: &dyn Fn(u32) -> u32,
) -> (bool, String) {
    let out = device
        .new_buffer(u64::from(threads) * 4, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    let group = threads.min(64);
    if let Err(e) = encoder.dispatch_threads(
        Size::new(threads.into(), 1, 1),
        Size::new(group.into(), 1, 1),
    ) {
        return (false, format!("dispatch refused: {e}"));
    }
    encoder.end_encoding().unwrap();
    if let Err(e) = command_buffer.commit() {
        return (false, format!("commit refused: {e}"));
    }
    if command_buffer.wait_until_completed_for(WAIT).is_err() {
        return (false, format!("no completion within {} s", WAIT.as_secs()));
    }
    if let Some(error) = command_buffer.error() {
        return (false, format!("command buffer error: {error:?}"));
    }
    let bytes = read_buffer(&out, 0, out.length()).unwrap();
    // Every thread of a 64-thread run, 64 spread over a wider one: the host's sums are O(words).
    let step = (threads / 64).max(1) as usize;
    let checked: Vec<u32> = (0..threads).step_by(step).collect();
    let wrong = checked
        .iter()
        .filter(|&&gid| {
            let at = gid as usize * 4;
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) != expected(gid)
        })
        .count();
    if wrong == 0 {
        (true, format!("correct ({} threads checked)", checked.len()))
    } else {
        (
            false,
            format!("WRONG in {wrong} of {} threads checked", checked.len()),
        )
    }
}

#[test]
fn sixteen_kib_per_thread_survives_wide_dispatches() {
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
    // Measure only the existing floor; enabling larger private arrays needs a separate sweep.
    for kib in [4, 8, 16] {
        let words = kib * 256;
        let state = state(&device, words).unwrap();
        for threads in [1, 64, 4096, 262_144] {
            let (correct, answer) = run(&device, &state, threads, &|gid| expected(words, gid));
            eprintln!(
                "private-memory on {}: {kib} KiB, {threads} threads: {answer}",
                device.name()
            );
            assert!(correct, "{kib} KiB, {threads} threads: {answer}");
        }
    }
    assert!(
        errors.lock().unwrap().is_empty(),
        "{:?}",
        errors.lock().unwrap()
    );
}
