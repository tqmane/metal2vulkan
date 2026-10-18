//! Authored fetch plus source-over on the native device, with no guest shaders.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan::metal::{BlendFactor, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CpuCacheMode, Device, DeviceKind, DeviceOptions, HazardTrackingMode,
    Library, LoadAction, MessageSeverity, MessageSink, Origin, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, Size, StorageMode, StoreAction, TextureDescriptor,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn library(device: &Device, name: &str) -> Library {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{name}.ll"));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&path).unwrap();
    device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    )
}

#[test]
fn normalized_source_over_fetches_each_earlier_draw_and_keeps_masked_channels() {
    let cache_path = std::env::temp_dir().join(format!("m2v-fetch-blend-{}", std::process::id()));
    std::fs::create_dir(&cache_path).unwrap();
    let cache = Scratch(cache_path);
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
    let fragment = device.compile_device().profile().fragment();
    eprintln!(
        "{}: ordered-inputs={}, pixel-interlock={}",
        device.name(),
        fragment.rasterization_order_color_access,
        fragment.pixel_interlock
    );
    let queue = device.new_command_queue().unwrap();
    let vertices = library(&device, "rt6_resolves");
    let fetch = library(&device, "fragment_fetch");
    let depth = device
        .new_buffer_with_bytes(&0.5f32.to_le_bytes(), ResourceOptions::shared())
        .unwrap();
    for pf in [PixelFormat::RGBA8Unorm, PixelFormat::BGRA8Unorm] {
        for mask in [
            ColorWriteMask::ALL,
            ColorWriteMask::RED | ColorWriteMask::BLUE,
        ] {
            let vertex = vertices.new_function_with_name("rt6_resolve_full").unwrap();
            let fetched = fetch.new_function_with_name("fetch_ordered").unwrap();
            let mut descriptor =
                RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
            descriptor.fragment = Some(StageSource::Function(fetched.function_descriptor()));
            let mut color = ColorAttachmentDescriptor::new(pf);
            color.blending_enabled = true;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
            color.write_mask = mask;
            descriptor.color_attachments[0] = Some(color);
            let mut unused = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
            unused.write_mask = ColorWriteMask::NONE;
            descriptor.color_attachments[1] = Some(unused);
            let pipeline = device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap();
            let mut texture = TextureDescriptor::texture_2d(pf, 4, 4, false);
            texture.usage = TextureUsage::RENDER_TARGET;
            texture.resource_options = ResourceOptions::private();
            let target = device.new_texture(&texture).unwrap();
            texture.pixel_format = PixelFormat::RGBA16Float;
            texture.resource_options = ResourceOptions::new(
                StorageMode::Memoryless,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            let unused = device.new_texture(&texture).unwrap();
            let mut pass = RenderPassDescriptor::new();
            pass.color_attachments[0].texture = Some(target.clone());
            pass.color_attachments[0].load_action = LoadAction::Clear;
            pass.color_attachments[0].store_action = StoreAction::Store;
            pass.color_attachments[0].clear_color =
                ClearColor::new(32.0 / 255.0, 64.0 / 255.0, 128.0 / 255.0, 0.0);
            pass.color_attachments[1].texture = Some(unused);
            pass.color_attachments[1].load_action = LoadAction::Clear;
            pass.color_attachments[1].store_action = StoreAction::DontCare;
            let command = queue.command_buffer().unwrap();
            let mut encoder = command.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            for _ in 0..3 {
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
            let readback = device.new_buffer(64, ResourceOptions::shared()).unwrap();
            let mut blit = command.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                &target,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(4, 4, 1),
                &readback,
                0,
                16,
                64,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command.commit().unwrap();
            command
                .wait_until_completed_for(Duration::from_secs(10))
                .unwrap();
            assert_eq!(command.error(), None);
            // Alpha stays zero: source-over multiplies each fetched color by 1.5 per draw.
            let green = if mask == ColorWriteMask::ALL { 216 } else { 64 };
            let rgba: [u8; 4] = [108, green, 255, 0];
            let bytes = if pf == PixelFormat::BGRA8Unorm {
                [rgba[2], rgba[1], rgba[0], rgba[3]]
            } else {
                rgba
            };
            // SAFETY: the readback remains alive and its GPU copy has completed.
            let actual =
                unsafe { std::slice::from_raw_parts(readback.contents().unwrap().as_ptr(), 64) };
            assert_eq!(actual, bytes.repeat(16), "{pf:?} {mask:?}");
        }
    }
    drop(depth);
    drop(fetch);
    drop(vertices);
    drop(queue);
    drop(device);
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "{errors:?}");
}
