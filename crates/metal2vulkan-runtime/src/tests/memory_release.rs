use super::*;
use ash::vk::Handle as _;
use std::sync::atomic::{AtomicBool, Ordering};

struct Context {
    memory: Mutex<Memory>,
    called: AtomicBool,
    unlocked: AtomicBool,
}

unsafe extern "system" fn free_memory(
    device: vk::Device,
    _: vk::DeviceMemory,
    _: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: the synthetic device handle points to the test's live boxed context.
    let context = unsafe { &*(device.as_raw() as usize as *const Context) };
    context
        .unlocked
        .store(context.memory.try_lock().is_ok(), Ordering::SeqCst);
    context.called.store(true, Ordering::SeqCst);
}

#[test]
fn native_block_release_does_not_hold_the_allocator_mutex() {
    let properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        ..Default::default()
    };
    let mut memory = Memory::new(&properties, BLOCK_BYTES, false, false, vec![0]);
    let block = Block {
        memory: vk::DeviceMemory::from_raw(1),
        size: 64,
        mapped: None,
        addresses: None,
    };
    let pool = (Holds::Buffers, 0);
    let index = memory.add_block(pool, block, 64, true, 64);
    let placement = memory.placement(pool, index, 0, 64, false);
    let context = Box::new(Context {
        memory: Mutex::new(memory),
        called: AtomicBool::new(false),
        unlocked: AtomicBool::new(false),
    });
    let handle = vk::Device::from_raw((&*context as *const Context) as usize as u64);
    // SAFETY: only vkFreeMemory is reached, and it is the mock above, not a driver call.
    let device = unsafe {
        ash::Device::load_with(
            |name| {
                if name == c"vkFreeMemory" {
                    free_memory as *const () as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                }
            },
            handle,
        )
    };
    free(&context.memory, &device, &placement);
    assert!(context.called.load(Ordering::SeqCst));
    assert!(
        context.unlocked.load(Ordering::SeqCst),
        "native release held the allocator mutex"
    );
}
