//! Native Windows host imports on the explicitly selected NVIDIA test device, without shaders.
#![cfg(all(windows, m2v_gpu))]

use ash::vk;
use metal2vulkan_runtime::{Device, DeviceKind, DeviceOptions, ResourceOptions};
use std::ffi::{c_char, c_void};
use std::ptr::NonNull;
use std::time::Duration;

const BYTES: usize = 8192;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(at: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(at: *mut c_void, size: usize, kind: u32) -> i32;
    fn CreateFileMappingW(
        file: *mut c_void,
        security: *const c_void,
        protect: u32,
        high: u32,
        low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFile(
        section: *mut c_void,
        access: u32,
        high: u32,
        low: u32,
        size: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(at: *const c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(
        module: *mut c_void,
        name: *const c_char,
    ) -> Option<unsafe extern "system" fn()>;
}

/// All copies complete before the caller may unmap either view.
fn round_trip(device: &Device, at: NonNull<u8>, observed: NonNull<u8>, writable: bool) {
    // SAFETY: the test owns committed pages until every resource below is released; the GPU
    // writes only the writable view, and the CPU reads only after completion.
    let imported = unsafe {
        device.new_buffer_with_bytes_no_copy(
            metal2vulkan_runtime::HostMemory::new(at),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let buffer = match imported {
        Ok(buffer) => buffer,
        Err(error) if !writable => {
            eprintln!("read-only Windows section import refused: {error:?}");
            let metal2vulkan_runtime::Error::Unsupported(unsupported) = error else {
                panic!("read-only view was not refused as host memory: {error:?}");
            };
            assert_eq!(
                unsupported.feature,
                metal2vulkan::error::Feature::HostMemory
            );
            return;
        }
        Err(error) => panic!("writable Windows import refused: {error:?}"),
    };
    assert_eq!(buffer.contents(), Some(at));
    // SAFETY: no GPU work has started, and the caller owns readable pages.
    let expected = unsafe { std::slice::from_raw_parts(at.as_ptr(), BYTES) }.to_vec();
    let readback = device
        .new_buffer(BYTES as u64, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let copy = queue.command_buffer().unwrap();
    let mut blit = copy.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&buffer, 0, &readback, 0, BYTES as u64)
        .unwrap();
    blit.end_encoding().unwrap();
    copy.commit().unwrap();
    copy.wait_until_completed_for(Duration::from_secs(10))
        .unwrap();
    copy.wait_until_completed().unwrap();
    // SAFETY: completed GPU writes; the readback buffer keeps all bytes mapped.
    let bytes = unsafe { std::slice::from_raw_parts(readback.contents().unwrap().as_ptr(), BYTES) };
    assert_eq!(bytes, expected);
    if writable {
        let fill = queue.command_buffer().unwrap();
        let mut blit = fill.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..BYTES as u64, 0xa6).unwrap();
        blit.end_encoding().unwrap();
        fill.commit().unwrap();
        fill.wait_until_completed_for(Duration::from_secs(10))
            .unwrap();
        fill.wait_until_completed().unwrap();
        // SAFETY: the GPU completed; this view maps the same pages and remains live.
        let bytes = unsafe { std::slice::from_raw_parts(observed.as_ptr(), BYTES) };
        assert!(bytes.iter().all(|&byte| byte == 0xa6));
    }
    eprintln!("Windows no-copy round trip: writable={writable}, bytes={BYTES}");
}

/// Reverse two section pages in adjacent 4 KiB views, as a guest's packed page mapping does.
unsafe fn scattered_view(section: *mut c_void) -> NonNull<u8> {
    type Allocate = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        usize,
        u32,
        u32,
        *const c_void,
        u32,
    ) -> *mut c_void;
    type Map = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        u64,
        usize,
        u32,
        u32,
        *const c_void,
        u32,
    ) -> *mut c_void;
    // SAFETY: KernelBase is loaded; the two exports have the documented Win32 signatures.
    let (allocate, map) = unsafe {
        let module = GetModuleHandleA(c"kernelbase.dll".as_ptr());
        assert!(!module.is_null());
        let allocate = GetProcAddress(module, c"VirtualAlloc2".as_ptr()).unwrap();
        let map = GetProcAddress(module, c"MapViewOfFile3".as_ptr()).unwrap();
        (
            std::mem::transmute::<unsafe extern "system" fn(), Allocate>(allocate),
            std::mem::transmute::<unsafe extern "system" fn(), Map>(map),
        )
    };
    // SAFETY: a fresh placeholder reservation, split into its two pages, then replaced by
    // shared views of the caller's section. The caller unmaps both once GPU work completes.
    unsafe {
        let at = NonNull::new(
            allocate(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                BYTES,
                0x42000,
                0x01,
                std::ptr::null(),
                0,
            )
            .cast::<u8>(),
        )
        .unwrap();
        assert_ne!(VirtualFree(at.as_ptr().cast(), BYTES / 2, 0x8002), 0);
        for (page, offset) in [4096, 0].into_iter().enumerate() {
            let target = at.as_ptr().add(page * 4096).cast();
            assert_eq!(
                map(
                    section,
                    std::ptr::null_mut(),
                    target,
                    offset,
                    4096,
                    0x4000,
                    0x04,
                    std::ptr::null(),
                    0
                ),
                target
            );
        }
        at
    }
}

#[test]
fn nvidia_windows_private_and_shared_pages_round_trip() {
    let mut options = DeviceOptions::default();
    options.kinds = vec![DeviceKind::Discrete];
    let device = Device::system_default_with(&options).unwrap();
    assert_eq!(device.name(), "NVIDIA GeForce RTX 4070 SUPER");
    // SAFETY: queries only the selected live device; no raw object is destroyed or submitted.
    let raw = unsafe { device.raw() };
    let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
    unsafe {
        raw.instance
            .get_physical_device_properties2(raw.physical_device, &mut properties)
    };
    assert_eq!(properties.properties.vendor_id, 0x10de);
    let usage: vk::BufferUsageFlags = device.vk_buffer_usage().into();
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let mut external = vk::ExternalBufferProperties::default();
    unsafe {
        raw.instance.get_physical_device_external_buffer_properties(
            raw.physical_device,
            &info,
            &mut external,
        )
    };
    let features = external.external_memory_properties.external_memory_features;
    assert!(features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE));
    assert!(!features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY));
    eprintln!(
        "{}: vendor={:#x}, device={:#x}, host alignment={}, external buffer features={}",
        device.name(),
        properties.properties.vendor_id,
        properties.properties.device_id,
        host.min_imported_host_pointer_alignment,
        features.as_raw()
    );
    // SAFETY: fresh committed private pages and section views, released after each round trip.
    unsafe {
        let private =
            NonNull::new(VirtualAlloc(std::ptr::null_mut(), BYTES, 0x3000, 0x04).cast::<u8>())
                .unwrap();
        std::ptr::write_bytes(private.as_ptr(), 0x17, BYTES);
        round_trip(&device, private, private, true);
        assert_ne!(VirtualFree(private.as_ptr().cast(), 0, 0x8000), 0);

        let section = CreateFileMappingW(
            -1_isize as *mut c_void,
            std::ptr::null(),
            0x04,
            0,
            BYTES as u32,
            std::ptr::null(),
        );
        assert!(!section.is_null());
        let write = NonNull::new(MapViewOfFile(section, 0x06, 0, 0, BYTES).cast::<u8>()).unwrap();
        let read = NonNull::new(MapViewOfFile(section, 0x04, 0, 0, BYTES).cast::<u8>()).unwrap();
        std::ptr::write_bytes(write.as_ptr(), 0x17, BYTES);
        round_trip(&device, write, read, true);
        round_trip(&device, read, read, false);
        std::ptr::write_bytes(write.as_ptr(), 0x11, BYTES / 2);
        std::ptr::write_bytes(write.as_ptr().add(BYTES / 2), 0x22, BYTES / 2);
        let packed = scattered_view(section);
        assert_eq!(*packed.as_ptr(), 0x22);
        assert_eq!(*packed.as_ptr().add(4096), 0x11);
        round_trip(&device, packed, read, true);
        assert_ne!(UnmapViewOfFile(packed.as_ptr().cast()), 0);
        assert_ne!(UnmapViewOfFile(packed.as_ptr().add(4096).cast()), 0);
        assert_ne!(UnmapViewOfFile(read.as_ptr().cast()), 0);
        assert_ne!(UnmapViewOfFile(write.as_ptr().cast()), 0);
        assert_ne!(CloseHandle(section), 0);
    }
}
