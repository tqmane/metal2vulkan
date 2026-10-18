//! Premultiplied source-over for a directly stored unsigned normalized attachment.
//!
//! The fetch and this blend run in the same ordered interlock. The source is clamped before
//! blending, as a normalized color attachment requires; the image store converts the result.
//! Arithmetic and conversion retain `DeviceColorConversion` rather than claiming Metal's bits.

use crate::lower::arith::{ops, FloatFmt, Form, Held};
use crate::metal::{BlendFactor, BlendOperation};
use crate::pipeline::lower_inputs::BlendEquation;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::BinOp;
use crate::vk;

pub(super) fn accepts(format: vk::Format, equation: BlendEquation) -> bool {
    format
        .block()
        .is_some_and(|b| b.color == Some(vk::NumericFormat::Unorm))
        && equation.source_rgb == BlendFactor::One
        && equation.destination_rgb == BlendFactor::OneMinusSourceAlpha
        && equation.rgb_operation == BlendOperation::Add
        && equation.source_alpha == BlendFactor::One
        && equation.destination_alpha == BlendFactor::OneMinusSourceAlpha
        && equation.alpha_operation == BlendOperation::Add
}

pub(super) fn emit(
    fb: &mut FunctionBuilder<'_>,
    source: Value,
    destination: Value,
) -> BResult<Value> {
    let form = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    };
    let zero = ops::fsplat(fb, form, 0.0)?;
    let one = ops::fsplat(fb, form, 1.0)?;
    // Ordered comparisons send a NaN to zero, independently of the driver's min/max rules.
    let positive = fb.binary(BinOp::FOrdGreaterThan, source, zero)?;
    let source = fb.select(positive, source, zero)?;
    let below_one = fb.binary(BinOp::FOrdLessThan, source, one)?;
    let source = fb.select(below_one, source, one)?;
    let alpha = fb.vector_shuffle(source, source, &[3, 3, 3, 3])?;
    let remaining = fb.binary_no_contraction(BinOp::FSub, one, alpha)?;
    let destination = fb.binary_no_contraction(BinOp::FMul, destination, remaining)?;
    fb.binary_no_contraction(BinOp::FAdd, source, destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
    use crate::spirv::builder::ModuleBuilder;
    use crate::spirv::interp::{self, Buffer, Config, Resources};
    use crate::spirv::ir::{
        natural_layout, ExecMode, ExecModel, LayoutRule, MemAccess, StorageClass, Type,
    };

    #[test]
    fn the_emitted_blend_clamps_the_source_and_uses_its_alpha() {
        let slot = ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        };
        for (source, destination, expected) in [
            (
                [0.25, 0.125, 0.0, 0.5],
                [0.5, 0.25, 1.0, 0.75],
                [0.5, 0.25, 0.5, 0.875],
            ),
            ([0.0; 4], [0.25, 0.5, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0]),
            ([1.0; 4], [0.25, 0.5, 0.75, 1.0], [1.0; 4]),
            (
                [-1.0, 2.0, f32::NAN, 0.25],
                [0.5; 4],
                [0.375, 1.375, 0.375, 0.625],
            ),
            (
                [0.25, 0.5, 0.75, f32::NAN],
                [0.25; 4],
                [0.5, 0.75, 1.0, 0.25],
            ),
        ] {
            let mut module = ModuleBuilder::new();
            let f32_ = module.ty(Type::Float { width: 32 }).unwrap();
            let vec4 = module.ty(Type::Vector { elem: f32_, len: 4 }).unwrap();
            let layout = natural_layout(module.types(), LayoutRule::Std430, true, &[vec4]).unwrap();
            let block = module
                .ty(Type::Struct {
                    members: [vec4].into(),
                    layout: Some(layout),
                })
                .unwrap();
            let output = module
                .resource(slot, StorageClass::StorageBuffer, block)
                .unwrap();
            let void = module.ty(Type::Void).unwrap();
            let mut function = module.function(void).unwrap();
            let scalar = Form {
                held: Held::Float(FloatFmt::F32),
                lanes: 1,
            };
            let mut vector = |values: [f32; 4]| {
                let parts =
                    values.map(|v| ops::fsplat(&mut function, scalar, f64::from(v)).unwrap());
                function.composite_construct(vec4, &parts).unwrap()
            };
            let source = vector(source);
            let destination = vector(destination);
            let result = emit(&mut function, source, destination).unwrap();
            let zero = function.u32_const(0).unwrap();
            let zero = function.module().const_value(zero).unwrap();
            let base = function.module().global_value(output).unwrap();
            let address = function.access_chain(base, &[zero]).unwrap();
            function
                .store(address, result, MemAccess::default())
                .unwrap();
            function.ret(None).unwrap();
            let main = function.finish().unwrap();
            module
                .entry(
                    ExecModel::GLCompute,
                    main,
                    vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
                )
                .unwrap();
            let module = module.finalize().unwrap();
            let resources = Resources {
                buffers: vec![Buffer {
                    slot: Some(slot),
                    address: None,
                    bytes: vec![0xAB; 16],
                }],
                ..Resources::default()
            };
            let (result, _) =
                interp::run_with_images(&module, &Config::default(), resources, Vec::new())
                    .unwrap();
            let actual: Vec<f32> = result.buffers[0]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual, expected);
        }
    }
}
