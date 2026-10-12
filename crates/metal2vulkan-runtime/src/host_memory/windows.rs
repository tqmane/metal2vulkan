//! Windows' host page size and committed, writable import windows: `GetSystemInfo` and
//! `VirtualQuery`. The driver still decides which pointers and memory types it can import.

use std::ffi::c_void;
use std::mem::MaybeUninit;

/// `SYSTEM_INFO` (sysinfoapi.h).
#[repr(C)]
struct SystemInfo {
    processor_architecture: u32,
    page_size: u32,
    minimum_application_address: *mut c_void,
    maximum_application_address: *mut c_void,
    active_processor_mask: usize,
    number_of_processors: u32,
    processor_type: u32,
    allocation_granularity: u32,
    processor_level: u16,
    processor_revision: u16,
}

/// `MEMORY_BASIC_INFORMATION` (winnt.h).
#[repr(C)]
struct MemoryBasicInformation {
    base_address: *const c_void,
    allocation_base: *const c_void,
    allocation_protect: u32,
    #[cfg(target_pointer_width = "64")]
    partition_id: u16,
    region_size: usize,
    state: u32,
    protect: u32,
    kind: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetSystemInfo(info: *mut SystemInfo);
    fn VirtualQuery(at: *const c_void, info: *mut MemoryBasicInformation, length: usize) -> usize;
}

/// Whether every byte lies in committed writable pages, including across adjacent section views.
pub(super) fn writable(start: u64, size: u64) -> bool {
    const MEM_COMMIT: u32 = 0x1000;
    const PAGE_GUARD: u32 = 0x100;
    // PAGE_READWRITE, WRITECOPY, EXECUTE_READWRITE, EXECUTE_WRITECOPY.
    const WRITABLE: u32 = 0x04 | 0x08 | 0x40 | 0x80;
    let Some(end) = start.checked_add(size) else {
        return false;
    };
    let (Ok(mut at), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
        return false;
    };
    while at < end {
        let mut info = MaybeUninit::<MemoryBasicInformation>::zeroed();
        let length = std::mem::size_of::<MemoryBasicInformation>();
        // SAFETY: VirtualQuery reads no bytes at the queried address and writes only the output.
        let got = unsafe { VirtualQuery(at as *const c_void, info.as_mut_ptr(), length) };
        if got != length {
            return false;
        }
        // SAFETY: the query filled the complete output.
        let info = unsafe { info.assume_init() };
        if info.state != MEM_COMMIT
            || info.protect & PAGE_GUARD != 0
            || info.protect & WRITABLE == 0
        {
            return false;
        }
        let base = info.base_address as usize;
        let Some(next) = base.checked_add(info.region_size) else {
            return false;
        };
        if base > at || next <= at {
            return false;
        }
        at = next;
    }
    true
}

/// The process's page size; allocation granularity is a separate, larger alignment.
pub(super) fn host_page() -> Option<u64> {
    let mut info = MaybeUninit::<SystemInfo>::zeroed();
    // SAFETY: GetSystemInfo writes one SYSTEM_INFO to the live, correctly sized output.
    let info = unsafe {
        GetSystemInfo(info.as_mut_ptr());
        info.assume_init()
    };
    (info.page_size != 0).then_some(u64::from(info.page_size))
}

const _: () = {
    assert!(std::mem::offset_of!(SystemInfo, page_size) == 4);
    assert!(std::mem::size_of::<SystemInfo>() == 24 + 3 * std::mem::size_of::<usize>());
    assert!(
        std::mem::size_of::<MemoryBasicInformation>()
            == if cfg!(target_pointer_width = "64") {
                48
            } else {
                28
            }
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(at: *mut c_void, length: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualProtect(at: *mut c_void, length: usize, protect: u32, old: *mut u32) -> i32;
        fn VirtualFree(at: *mut c_void, length: usize, kind: u32) -> i32;
    }

    #[test]
    fn the_query_accepts_only_committed_writable_pages() {
        let page = host_page().unwrap() as usize;
        // SAFETY: a fresh two-page private allocation, released below.
        let at = unsafe { VirtualAlloc(std::ptr::null_mut(), 2 * page, 0x3000, 0x04) };
        assert!(!at.is_null());
        let start = at as u64;
        assert!(writable(start, 2 * page as u64));
        let mut old = 0;
        // SAFETY: changes only this allocation's second page.
        assert_ne!(
            unsafe { VirtualProtect(at.add(page), page, 0x02, &mut old) },
            0
        );
        assert!(!writable(start, 2 * page as u64));
        // SAFETY: as above, now inaccessible, then guarded.
        assert_ne!(
            unsafe { VirtualProtect(at.add(page), page, 0x01, &mut old) },
            0
        );
        assert!(!writable(start, 2 * page as u64));
        assert_ne!(
            unsafe { VirtualProtect(at.add(page), page, 0x104, &mut old) },
            0
        );
        assert!(!writable(start, 2 * page as u64));
        // SAFETY: decommit its second page; the reserved address cannot be reused meanwhile.
        assert_ne!(unsafe { VirtualFree(at.add(page), page, 0x4000) }, 0);
        assert!(writable(start, page as u64));
        assert!(!writable(start, 2 * page as u64));
        assert!(!writable(u64::MAX, page as u64));
        // SAFETY: releases the reservation this test owns, including the committed first page.
        assert_ne!(unsafe { VirtualFree(at, 0, 0x8000) }, 0);
    }
}
