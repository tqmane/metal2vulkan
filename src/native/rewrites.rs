//! SPIR-V rewrite entry points: each applies one legalization/portability rewrite from a sibling
//! pass module (`psb`, `phi_index`, `cfg`, `relooper`, `constfold`, …) to an in-flight [`Module`].
//! Callers (the failure-triggered retry tiers in `lib.rs`) adopt a result only if it independently
//! validates, so every rewrite here is floor-safe by construction. Also holds the remaining
//! structural screens (`has_*`); public byte compatibility wrappers live at the `native` facade.

use super::*;

/// Whether `spv` contains the local selection emitted to guard a raw logical-buffer write whose
/// source byte offset has a dynamic term wider than the u32 address model.  The guard's true arm
/// performs the original write and its false arm falls through to the selection merge, preserving
/// Metal's robust no-write behavior when the complete i64 offset cannot be represented.
///
/// This deliberately identifies the emitted control-flow *shape*, not a source symbol or private capture
/// case.  A guard inserted into a source loop header moves that header's source terminator into the
/// guard continuation, which can expose a CFG-only validator error.  Callers can then reuse the
/// ordinary relooper on precisely that already-guarded graph; the resulting bytes are the same
/// candidate production's first CFG retry would otherwise adopt.
pub(crate) fn module_has_wide_raw_store_guard(module: &Module) -> bool {
    let defs: HashMap<Word, &Instruction> = module
        .all_inst_iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst)))
        .collect();

    let is_u64_constant = |id: Word| {
        let Some(inst) = defs.get(&id) else {
            return false;
        };
        if inst.class.opcode != Op::Constant {
            return false;
        }
        let Some(ty) = inst.result_type else {
            return false;
        };
        let Some(type_inst) = defs.get(&ty) else {
            return false;
        };
        type_inst.class.opcode == Op::TypeInt
            && type_inst.operands.first() == Some(&Operand::LiteralBit32(64))
            && type_inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
    };

    for function in &module.functions {
        let blocks: HashMap<Word, &crate::spirv_module::Block> = function
            .blocks
            .iter()
            .filter_map(|block| {
                block
                    .label
                    .as_ref()
                    .and_then(|label| label.result_id)
                    .map(|id| (id, block))
            })
            .collect();
        for block in &function.blocks {
            for (index, branch) in block.instructions.iter().enumerate() {
                if branch.class.opcode != Op::BranchConditional || index == 0 {
                    continue;
                }
                let (
                    Some(Operand::IdRef(condition)),
                    Some(Operand::IdRef(write_label)),
                    Some(Operand::IdRef(false_label)),
                ) = (
                    branch.operands.first(),
                    branch.operands.get(1),
                    branch.operands.get(2),
                )
                else {
                    continue;
                };
                let merge = &block.instructions[index - 1];
                if merge.class.opcode != Op::SelectionMerge {
                    continue;
                }
                let Some(Operand::IdRef(merge_label)) = merge.operands.first() else {
                    continue;
                };
                if false_label != merge_label {
                    continue;
                }
                let Some(compare) = defs.get(condition) else {
                    continue;
                };
                if compare.class.opcode != Op::ULessThanEqual
                    || !matches!(compare.operands.get(1), Some(Operand::IdRef(max)) if is_u64_constant(*max))
                {
                    continue;
                }
                let Some(write_block) = blocks.get(write_label) else {
                    continue;
                };
                let writes_then_merges = write_block
                    .instructions
                    .iter()
                    .any(|inst| is_spirv_memory_write(inst.class.opcode))
                    && write_block.instructions.last().is_some_and(|terminator| {
                        terminator.class.opcode == Op::Branch
                            && terminator.operands.first() == Some(&Operand::IdRef(*merge_label))
                    });
                if writes_then_merges {
                    return true;
                }
            }
        }
    }
    false
}

/// `OpStore` plus the atomic operations that can modify their first pointer operand.  The native
/// raw subword-store lowering is an atomic AND/OR read-modify-write, so the robust wide-offset
/// guard must recognize it as a write too.
fn is_spirv_memory_write(op: Op) -> bool {
    matches!(
        op,
        Op::Store
            | Op::AtomicStore
            | Op::AtomicExchange
            | Op::AtomicCompareExchange
            | Op::AtomicCompareExchangeWeak
            | Op::AtomicIIncrement
            | Op::AtomicIDecrement
            | Op::AtomicIAdd
            | Op::AtomicISub
            | Op::AtomicSMin
            | Op::AtomicUMin
            | Op::AtomicSMax
            | Op::AtomicUMax
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor
            | Op::AtomicFAddEXT
            | Op::AtomicFMinEXT
            | Op::AtomicFMaxEXT
    )
}

/// Drop byte-addressed Workgroup padding clears that the native emitter can produce from AIR
/// `llvm.memset.p3` over struct tail padding.
///
/// SPIR-V Logical addressing cannot express `OpPtrAccessChain %uchar*` from a `%struct*` base. When
/// the byte offset does not match any `Offset`-decorated struct member and every use is a zero
/// `OpStore`, the store only clears padding. Workgroup memory has already been zero-filled by the
/// harness prologue, and no typed load can observe the padding byte, so removing the invalid chain
/// and its zero stores is value-preserving.
pub(crate) fn drop_workgroup_struct_padding_byte_zero_stores_module(module: &mut Module) -> bool {
    use spirv::Decoration;

    let mut type_defs: HashMap<Word, &Instruction> = HashMap::new();
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    let mut result_type: HashMap<Word, Word> = HashMap::new();
    let mut constants: HashMap<Word, u64> = HashMap::new();
    let mut member_offsets: HashMap<Word, HashMap<u32, u32>> = HashMap::new();

    for inst in module.all_inst_iter() {
        if let Some(id) = inst.result_id {
            if let Some(ty) = inst.result_type {
                result_type.insert(id, ty);
            }
            match inst.class.opcode {
                Op::TypeInt | Op::TypePointer | Op::TypeStruct => {
                    type_defs.insert(id, inst);
                }
                Op::Constant => {
                    let value = match inst.operands.first() {
                        Some(Operand::LiteralBit32(value)) => u64::from(*value),
                        Some(Operand::LiteralBit64(value)) => *value,
                        _ => continue,
                    };
                    constants.insert(id, value);
                }
                Op::ConstantNull => {
                    constants.insert(id, 0);
                }
                _ => {}
            }
        }
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(sc)), Some(Operand::IdRef(pointee))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*sc, *pointee));
            }
        }
        if inst.class.opcode == Op::MemberDecorate {
            let (
                Some(Operand::IdRef(struct_ty)),
                Some(Operand::LiteralBit32(member)),
                Some(Operand::Decoration(Decoration::Offset)),
                Some(Operand::LiteralBit32(offset)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
                inst.operands.get(3),
            )
            else {
                continue;
            };
            member_offsets
                .entry(*struct_ty)
                .or_default()
                .insert(*member, *offset);
        }
    }

    let is_uchar = |ty: Word| -> bool {
        matches!(
            type_defs.get(&ty).map(|inst| inst.operands.as_slice()),
            Some([Operand::LiteralBit32(8), Operand::LiteralBit32(0)])
        )
    };
    fn align_to(value: u32, align: u32) -> u32 {
        if align <= 1 {
            value
        } else {
            value.div_ceil(align) * align
        }
    }
    fn type_size_align(
        type_defs: &HashMap<Word, &Instruction>,
        constants: &HashMap<Word, u64>,
        ty: Word,
    ) -> Option<(u32, u32)> {
        let def = type_defs.get(&ty)?;
        match def.class.opcode {
            Op::TypeInt | Op::TypeFloat => {
                let Some(Operand::LiteralBit32(width)) = def.operands.first() else {
                    return None;
                };
                let bytes = width.div_ceil(8).max(1);
                Some((bytes, bytes))
            }
            Op::TypeVector => {
                let (Some(Operand::IdRef(component)), Some(Operand::LiteralBit32(count))) =
                    (def.operands.first(), def.operands.get(1))
                else {
                    return None;
                };
                let (size, align) = type_size_align(type_defs, constants, *component)?;
                Some((size.checked_mul(*count)?, align))
            }
            Op::TypeArray => {
                let (Some(Operand::IdRef(element)), Some(Operand::IdRef(length))) =
                    (def.operands.first(), def.operands.get(1))
                else {
                    return None;
                };
                let (size, align) = type_size_align(type_defs, constants, *element)?;
                let count = u32::try_from(*constants.get(length)?).ok()?;
                Some((align_to(size, align).checked_mul(count)?, align))
            }
            Op::TypeStruct => {
                let mut offset = 0u32;
                let mut max_align = 1u32;
                for operand in &def.operands {
                    let Operand::IdRef(member_ty) = operand else {
                        return None;
                    };
                    let (member_size, member_align) =
                        type_size_align(type_defs, constants, *member_ty)?;
                    max_align = max_align.max(member_align);
                    offset = align_to(offset, member_align);
                    offset = offset.checked_add(member_size)?;
                }
                Some((align_to(offset, max_align), max_align))
            }
            _ => None,
        }
    }
    let inferred_struct_offsets = |ty: Word| -> Option<HashSet<u32>> {
        let def = type_defs.get(&ty)?;
        if def.class.opcode != Op::TypeStruct {
            return None;
        }
        let mut offsets = HashSet::new();
        let mut offset = 0u32;
        for operand in &def.operands {
            let Operand::IdRef(member_ty) = operand else {
                return None;
            };
            let (member_size, member_align) = type_size_align(&type_defs, &constants, *member_ty)?;
            offset = align_to(offset, member_align);
            offsets.insert(offset);
            offset = offset.checked_add(member_size)?;
        }
        Some(offsets)
    };
    let is_struct_padding_offset = |ty: Word, offset: u32| -> bool {
        let Some(def) = type_defs.get(&ty) else {
            return false;
        };
        if def.class.opcode != Op::TypeStruct {
            return false;
        }
        if let Some(offsets) = member_offsets.get(&ty) {
            return !offsets
                .values()
                .any(|member_offset| *member_offset == offset);
        };
        inferred_struct_offsets(ty).is_some_and(|offsets| !offsets.contains(&offset))
    };

    let mut chain_base: HashMap<Word, (usize, usize, usize, Word, u32)> = HashMap::new();
    for (fi, function) in module.functions.iter().enumerate() {
        for (bi, block) in function.blocks.iter().enumerate() {
            for (ii, inst) in block.instructions.iter().enumerate() {
                if inst.class.opcode != Op::PtrAccessChain || inst.operands.len() != 2 {
                    continue;
                }
                let (Some(chain), Some(result_ptr_ty)) = (inst.result_id, inst.result_type) else {
                    continue;
                };
                let Some(&(StorageClass::Workgroup, result_pointee)) = ptr_info.get(&result_ptr_ty)
                else {
                    continue;
                };
                if !is_uchar(result_pointee) {
                    continue;
                }
                let (Some(Operand::IdRef(base)), Some(Operand::IdRef(offset_id))) =
                    (inst.operands.first(), inst.operands.get(1))
                else {
                    continue;
                };
                let Some(offset) = constants
                    .get(offset_id)
                    .and_then(|value| u32::try_from(*value).ok())
                else {
                    continue;
                };
                let Some(base_ptr_ty) = result_type.get(base).copied() else {
                    continue;
                };
                let Some(&(StorageClass::Workgroup, base_pointee)) = ptr_info.get(&base_ptr_ty)
                else {
                    continue;
                };
                if is_struct_padding_offset(base_pointee, offset) {
                    chain_base.insert(chain, (fi, bi, ii, base_pointee, offset));
                }
            }
        }
    }
    let mut changed_candidates = true;
    while changed_candidates {
        changed_candidates = false;
        for (fi, function) in module.functions.iter().enumerate() {
            for (bi, block) in function.blocks.iter().enumerate() {
                for (ii, inst) in block.instructions.iter().enumerate() {
                    if inst.class.opcode != Op::PtrAccessChain || inst.operands.len() != 2 {
                        continue;
                    }
                    let (Some(chain), Some(result_ptr_ty)) = (inst.result_id, inst.result_type)
                    else {
                        continue;
                    };
                    if chain_base.contains_key(&chain) {
                        continue;
                    }
                    let Some(&(StorageClass::Workgroup, result_pointee)) =
                        ptr_info.get(&result_ptr_ty)
                    else {
                        continue;
                    };
                    if !is_uchar(result_pointee) {
                        continue;
                    }
                    let (Some(Operand::IdRef(base)), Some(Operand::IdRef(offset_id))) =
                        (inst.operands.first(), inst.operands.get(1))
                    else {
                        continue;
                    };
                    let Some((_, _, _, root_struct, base_offset)) = chain_base.get(base).copied()
                    else {
                        continue;
                    };
                    let Some(offset) = constants
                        .get(offset_id)
                        .and_then(|value| u32::try_from(*value).ok())
                        .and_then(|offset| base_offset.checked_add(offset))
                    else {
                        continue;
                    };
                    if is_struct_padding_offset(root_struct, offset) {
                        chain_base.insert(chain, (fi, bi, ii, root_struct, offset));
                        changed_candidates = true;
                    }
                }
            }
        }
    }
    if chain_base.is_empty() {
        return false;
    }

    let mut zero_values: HashSet<Word> = constants
        .iter()
        .filter_map(|(id, value)| (*value == 0).then_some(*id))
        .collect();
    let mut changed_zero = true;
    while changed_zero {
        changed_zero = false;
        for inst in module.all_inst_iter() {
            let Some(id) = inst.result_id else {
                continue;
            };
            if zero_values.contains(&id) {
                continue;
            }
            let zero_operand0 = inst.operands.first().is_some_and(
                |operand| matches!(operand, Operand::IdRef(value) if zero_values.contains(value)),
            );
            let zero = match inst.class.opcode {
                Op::CopyObject | Op::UConvert | Op::SConvert | Op::Bitcast => zero_operand0,
                Op::ShiftRightLogical | Op::ShiftRightArithmetic | Op::ShiftLeftLogical => {
                    zero_operand0
                }
                _ => false,
            };
            if zero {
                zero_values.insert(id);
                changed_zero = true;
            }
        }
    }

    let mut store_sites: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut disqualified: HashSet<Word> = HashSet::new();
    for (fi, function) in module.functions.iter().enumerate() {
        for (bi, block) in function.blocks.iter().enumerate() {
            for (ii, inst) in block.instructions.iter().enumerate() {
                for (oi, operand) in inst.operands.iter().enumerate() {
                    let Operand::IdRef(id) = operand else {
                        continue;
                    };
                    if !chain_base.contains_key(id) {
                        continue;
                    }
                    if inst.class.opcode == Op::PtrAccessChain
                        && oi == 0
                        && inst
                            .result_id
                            .is_some_and(|result| chain_base.contains_key(&result))
                    {
                        continue;
                    }
                    if inst.class.opcode == Op::Store
                        && oi == 0
                        && inst.operands.get(1).and_then(|operand| match operand {
                            Operand::IdRef(value) => zero_values.contains(value).then_some(0),
                            _ => None,
                        }) == Some(0)
                    {
                        store_sites.insert((fi, bi, ii));
                    } else {
                        disqualified.insert(*id);
                    }
                }
            }
        }
    }
    chain_base.retain(|chain, _| !disqualified.contains(chain));
    if chain_base.is_empty() || store_sites.is_empty() {
        return false;
    }

    let chain_sites: HashSet<(usize, usize, usize)> = chain_base
        .into_values()
        .map(|(fi, bi, ii, _, _)| (fi, bi, ii))
        .collect();
    let mut changed = false;
    for (fi, function) in module.functions.iter_mut().enumerate() {
        for (bi, block) in function.blocks.iter_mut().enumerate() {
            let old = std::mem::take(&mut block.instructions);
            block.instructions = old
                .into_iter()
                .enumerate()
                .filter_map(|(ii, inst)| {
                    if chain_sites.contains(&(fi, bi, ii)) || store_sites.contains(&(fi, bi, ii)) {
                        changed = true;
                        None
                    } else {
                        Some(inst)
                    }
                })
                .collect();
        }
    }
    changed
}

/// The type of the member `path` reaches inside `composite`, or `None` when the
/// path leaves a type this walk models.
///
/// Only the aggregates a composite insert/extract can index are followed —
/// struct members by literal index, and the uniform element of an array, vector
/// or matrix. An index past the end of a struct, or into anything else, gives up
/// rather than guessing, because the caller uses the answer to decide whether a
/// write is ill-typed and a wrong answer there would delete a real one.
fn member_type_at(
    types: &HashMap<Word, Instruction>,
    composite: Word,
    path: &[u32],
) -> Option<Word> {
    let mut current = composite;
    for index in path {
        let inst = types.get(&current)?;
        current = match inst.class.opcode {
            Op::TypeStruct => match inst.operands.get(*index as usize)? {
                Operand::IdRef(member) => *member,
                _ => return None,
            },
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match inst.operands.first()? {
                    Operand::IdRef(element) => *element,
                    _ => return None,
                }
            }
            _ => return None,
        };
    }
    Some(current)
}

/// Forward opaque handles out of the composites the AIR calling convention
/// bundles them into, so no `OpCompositeInsert` ever names one.
///
/// Vulkan is explicit that a handle cannot live in a composite —
/// `VUID-StandaloneSpirv-None-04667`, "OpTypeStruct must not contain an invalid
/// opaque type" — so the bundle's members are modelled as byte pointers and the
/// real handle is inserted into them anyway. spirv-val rejects that insert: "The
/// Object type (OpTypeImage) does not match the type that results from indexing
/// into the Composite (OpTypePointer)". Retyping the member is not an escape;
/// the type itself would then be the illegal thing, which a minimal repro
/// confirms.
///
/// The shape is a resource bundle assembled, carried across a branch, and taken
/// apart again — inserted at depth two, read back at depth one and then one more:
///
///     %212 = OpCompositeInsert %bundle %101 %undef 0 0   ; %101 = OpLoad image
///     %213 = OpCompositeInsert %bundle %99  %212  1 0    ; %99  = OpLoad sampler
///     %217 = OpPhi %bundle %215 %211
///     %218 = OpCompositeExtract %wrapper %217 0
///     %222 = OpCompositeExtract %image %218 0
///
/// Every read of an opaque member is resolved back to the value that was written
/// and replaced with it directly. The insert chain then has no live consumer and
/// is removed, leaving the byte-pointer members untouched and unread — which is
/// legal, because the illegal part was only ever the write.
///
/// This is what stops CoreImage's separable-Gaussian passes —
/// `ci_asgDownH_affine_nearest` and its vertical counterpart, the backdrop blur
/// behind macOS vibrancy — from translating at all.
///
/// Deliberately conservative. A read is forwarded only when it resolves to one
/// value on every path: through a phi all incomings must agree, and a partial
/// write over the region being read gives up. Only opaque-typed reads are
/// touched, so a module that already validates is unchanged.
pub(crate) fn forward_opaque_composite_members_module(module: &mut Module) -> bool {
    use std::collections::HashMap;

    let opaque_types: HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|inst| {
            matches!(
                inst.class.opcode,
                Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage
            )
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    if opaque_types.is_empty() {
        return false;
    }

    /// Resolve `extract(composite, path)` to the single value written there, or
    /// `None` when no single value answers on every path.
    fn resolve(
        composite: Word,
        path: &[u32],
        inserts: &HashMap<Word, (Word, Word, Vec<u32>)>,
        phis: &HashMap<Word, Vec<Word>>,
        depth: u32,
    ) -> Option<Word> {
        if depth > 32 {
            return None;
        }
        if path.is_empty() {
            return Some(composite);
        }
        if let Some((object, base, written)) = inserts.get(&composite) {
            if written.as_slice() == path {
                return Some(*object);
            }
            if path.starts_with(written.as_slice()) {
                return resolve(*object, &path[written.len()..], inserts, phis, depth + 1);
            }
            if written.starts_with(path) {
                // A partial write over the region being read: not one value.
                return None;
            }
            return resolve(*base, path, inserts, phis, depth + 1);
        }
        if let Some(incoming) = phis.get(&composite) {
            let mut answer: Option<Word> = None;
            for value in incoming {
                let resolved = resolve(*value, path, inserts, phis, depth + 1)?;
                match answer {
                    None => answer = Some(resolved),
                    Some(prev) if prev == resolved => {}
                    Some(_) => return None,
                }
            }
            return answer;
        }
        None
    }

    // Type declarations, so a write's member type can be checked against what it
    // stores. Collected once; `module.functions` is borrowed mutably below.
    let module_types: HashMap<Word, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
        .collect();

    let mut changed = false;
    for function in module.functions.iter_mut() {
        let mut inserts: HashMap<Word, (Word, Word, Vec<u32>)> = HashMap::new();
        let mut phis: HashMap<Word, Vec<Word>> = HashMap::new();
        let mut extracts: HashMap<Word, (Word, Vec<u32>)> = HashMap::new();
        // Seeded with the module scope: a bundle member is often a global
        // `Private` variable, whose result type lives in `types_global_values`
        // and not in any block. Without it the ill-typed-write check silently
        // skips exactly the writes it exists to find.
        let mut result_type: HashMap<Word, Word> = module_types
            .values()
            .filter_map(|inst| Some((inst.result_id?, inst.result_type?)))
            .collect();

        for block in &function.blocks {
            for inst in &block.instructions {
                let Some(rid) = inst.result_id else { continue };
                if let Some(rty) = inst.result_type {
                    result_type.insert(rid, rty);
                }
                match inst.class.opcode {
                    Op::CompositeInsert => {
                        let (Some(Operand::IdRef(object)), Some(Operand::IdRef(base))) =
                            (inst.operands.first(), inst.operands.get(1))
                        else {
                            continue;
                        };
                        let path: Vec<u32> = inst.operands[2..]
                            .iter()
                            .filter_map(|op| match op {
                                Operand::LiteralBit32(v) => Some(*v),
                                _ => None,
                            })
                            .collect();
                        if path.len() + 2 == inst.operands.len() {
                            inserts.insert(rid, (*object, *base, path));
                        }
                    }
                    Op::CompositeExtract => {
                        let Some(Operand::IdRef(base)) = inst.operands.first() else {
                            continue;
                        };
                        let path: Vec<u32> = inst.operands[1..]
                            .iter()
                            .filter_map(|op| match op {
                                Operand::LiteralBit32(v) => Some(*v),
                                _ => None,
                            })
                            .collect();
                        if path.len() + 1 == inst.operands.len() {
                            extracts.insert(rid, (*base, path));
                        }
                    }
                    Op::Phi => {
                        let mut incoming = Vec::new();
                        let mut k = 0;
                        while k + 1 < inst.operands.len() {
                            if let Operand::IdRef(v) = &inst.operands[k] {
                                incoming.push(*v);
                            }
                            k += 2;
                        }
                        phis.insert(rid, incoming);
                    }
                    _ => {}
                }
            }
        }

        // Extract-of-extract is one read at a joined path, which is how a
        // two-level bundle is taken apart.
        let flattened: HashMap<Word, (Word, Vec<u32>)> = extracts
            .iter()
            .map(|(rid, (base, path))| {
                let mut base = *base;
                let mut path = path.clone();
                for _ in 0..32 {
                    let Some((outer_base, outer_path)) = extracts.get(&base) else {
                        break;
                    };
                    let mut combined = outer_path.clone();
                    combined.extend_from_slice(&path);
                    base = *outer_base;
                    path = combined;
                }
                (*rid, (base, path))
            })
            .collect();

        let mut replace: HashMap<Word, Word> = HashMap::new();
        for (rid, (base, path)) in &flattened {
            // Keyed on what the read RESOLVES to, not on how the read is typed.
            // The bundle's members are declared as `UniformConstant` byte
            // pointers, so the read of a handle is declared that way too:
            //
            //     %255 = OpCompositeExtract %_ptr_UniformConstant_uchar %216 0
            //     %257 = OpSampledImage %86 %255 %256
            //
            // — a byte pointer handed straight to `OpSampledImage` as its image.
            // Testing the read's own type would miss every one of these, which is
            // the whole class. The written value is the real handle, and putting
            // it at the use makes both the use and the module well-typed.
            let Some(value) = resolve(*base, path, &inserts, &phis, 0) else {
                continue;
            };
            let resolves_to_handle = result_type
                .get(&value)
                .is_some_and(|ty| opaque_types.contains(ty));
            if resolves_to_handle && value != *rid {
                replace.insert(*rid, value);
            }
        }
        if replace.is_empty() && inserts.is_empty() {
            continue;
        }

        // A handle written where nothing reads it.
        //
        // The same bundle appears in two shapes depending on which tier produced
        // the module: in one the handle's slot is read back and handed to
        // `OpSampledImage` — forwarded above — and in the other nothing reads it
        // at all, because the call the bundle was assembled for has been inlined
        // away. An unread write has no value to forward to a use, and the only
        // thing it accomplishes is to make the module invalid, so it goes.
        //
        // Decided by reachability, not by whether the forwarding above happened
        // to succeed.
        //
        // `resolve` returns `None` for "cannot tell" as well as for "nothing
        // there" — a phi whose incomings disagree, a partial write over the
        // region, a base this walk does not model. Reading that as "no reader"
        // and deleting the write is unsound, and it is not a theoretical
        // unsoundness: a handle removed that way leaves the shader sampling an
        // undefined image, which took the host process down at the dispatch.
        //
        // So the question asked is the conservative one: can any read reach this
        // write at all? Walk back from every extract's base through insert bases
        // and phi incomings; a write whose result is not in that set has no
        // reader on any path and only then goes.
        let mut reachable: HashSet<Word> = HashSet::new();
        let mut pending: Vec<Word> = flattened.values().map(|(base, _)| *base).collect();
        while let Some(id) = pending.pop() {
            if !reachable.insert(id) {
                continue;
            }
            if let Some((object, base, _)) = inserts.get(&id) {
                pending.push(*base);
                pending.push(*object);
            }
            if let Some(incoming) = phis.get(&id) {
                pending.extend(incoming.iter().copied());
            }
        }
        for (rid, (object, base, _)) in &inserts {
            if replace.contains_key(rid) {
                continue;
            }
            let writes_handle = result_type
                .get(object)
                .is_some_and(|ty| opaque_types.contains(ty));
            if writes_handle && !reachable.contains(rid) {
                replace.insert(*rid, *base);
            }
        }
        if replace.is_empty() {
            continue;
        }

        // Chained writes replace to each other — dropping `insert(h, %780)` and
        // `insert(h, %288)` maps 781 to 780 and 780 to 288, and 780 is itself
        // going away. Fold each entry to its final destination before rewriting,
        // or the use is left naming a deleted id.
        let keys: Vec<Word> = replace.keys().copied().collect();
        for key in keys {
            let mut value = replace[&key];
            for _ in 0..64 {
                match replace.get(&value) {
                    Some(next) if *next != value => value = *next,
                    _ => break,
                }
            }
            replace.insert(key, value);
        }

        for block in function.blocks.iter_mut() {
            for inst in block.instructions.iter_mut() {
                for operand in inst.operands.iter_mut() {
                    if let Operand::IdRef(id) = operand {
                        if let Some(value) = replace.get(id) {
                            *operand = Operand::IdRef(*value);
                            changed = true;
                        }
                    }
                }
            }
            block
                .instructions
                .retain(|inst| !inst.result_id.is_some_and(|rid| replace.contains_key(&rid)));
        }

        // The inserts that named a handle now have no consumer. Drop them and
        // whatever else in the chain went dead with them; bounded rounds, since
        // each round only removes.
        for _ in 0..16 {
            let used: HashSet<Word> = function
                .blocks
                .iter()
                .flat_map(|b| b.instructions.iter())
                .flat_map(|inst| inst.operands.iter())
                .filter_map(|op| match op {
                    Operand::IdRef(id) => Some(*id),
                    _ => None,
                })
                .collect();
            let mut removed = false;
            for block in function.blocks.iter_mut() {
                let before = block.instructions.len();
                block.instructions.retain(|inst| {
                    let dead = matches!(
                        inst.class.opcode,
                        Op::CompositeInsert | Op::CompositeExtract | Op::Phi
                    ) && inst.result_id.is_some_and(|rid| !used.contains(&rid));
                    !dead
                });
                if block.instructions.len() != before {
                    removed = true;
                    changed = true;
                }
            }
            if !removed {
                break;
            }
        }
    }
    changed
}

/// Remove debug/decorate records whose target id was deleted by a prior module rewrite.
///
/// Rewrites such as constant-branch pruning can delete function-constant helper globals, and a later
/// CFG rebuild may preserve their `OpName` records. `OpName` is non-semantic, but SPIR-V still
/// requires its target id to be defined. This cleanup never touches executable instructions or
/// interface declarations; it only drops annotations/debug names that already point at nothing.
pub(crate) fn drop_dangling_debug_targets_module(module: &mut Module) -> bool {
    let defined: HashSet<Word> = module
        .all_inst_iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    let target_is_dangling = |inst: &Instruction| -> bool {
        matches!(inst.operands.first(), Some(Operand::IdRef(id)) if !defined.contains(id))
    };

    let before_names = module.debug_names.len();
    let before_annotations = module.annotations.len();
    module.debug_names.retain(|inst| !target_is_dangling(inst));
    module.annotations.retain(|inst| !target_is_dangling(inst));
    module.debug_names.len() != before_names || module.annotations.len() != before_annotations
}

/// Apply the W1 PhysicalStorageBuffer64 lowering in place. Errors if no cross-binding pointer-merge
/// sub-graph was rewritten. The caller (the failure-triggered retry) adopts the result ONLY if it
/// independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_cross_binding_pointer_merges_module(
    module: &mut Module,
) -> Result<(), String> {
    if !psb::rewrite_cross_binding_pointer_merges(module) {
        return Err("native emitter: no cross-binding pointer merge to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Report whether `module` has a lowerable cross-binding pointer closure containing an `OpPhi`.
/// Callers use this cheap structural screen after the exact spirv-val diagnostic to decide whether
/// the phi could be the failure that warrants the PhysicalStorageBuffer64 primary candidate.
pub(crate) fn has_cross_binding_pointer_phi_module(module: &Module) -> bool {
    psb::has_cross_binding_pointer_phi(module)
}

/// Apply the PhysicalStorageBuffer64 lowering in place only when the cross-binding closure contains
/// an `OpPhi`. Ordinary cross-binding selects stay available to the Logical value-domain lowering;
/// a phi with post-merge dynamic accesses needs the address-table representation instead of
/// replaying values on predecessor edges. The caller still validates before adopting the module.
pub(crate) fn rewrite_cross_binding_pointer_phis_module(module: &mut Module) -> Result<(), String> {
    if !psb::rewrite_cross_binding_pointer_phis(module) {
        return Err("native emitter: no cross-binding pointer phi to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Lower the cross-binding pointer-merge sub-graph in place INTO THE VALUE DOMAIN (plain Logical
/// `StorageBuffer`), staying off PhysicalStorageBuffer64. Instead of selecting among POINTERS then
/// loading once, it loads from every candidate buffer and selects among the LOADED VALUES —
/// byte-exact (the selected value is the exact load Apple performs; discarded over-reads do not
/// fault), and MoltenVK-runnable (no buffer-device-address, which blocks compute-pipeline creation).
/// Errors if no cross-binding pointer merge was value-lowered. The caller (the failure-triggered
/// retry) adopts the result ONLY if it independently validates, so this is floor-safe by
/// construction; it is preferred over the PSB lowering when both validate.
pub(crate) fn rewrite_cross_binding_pointer_merges_to_values_module(
    module: &mut Module,
) -> Result<(), String> {
    if !psb_value_select::rewrite_cross_binding_pointer_merges_to_values(module) {
        return Err("native emitter: no cross-binding pointer merge to value-lower".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Lower an opaque image `OpSelect` through pure explicit-LOD sampling into ordinary value selects,
/// in place. Vulkan cannot select images directly without descriptor indexing, while cloning a pure
/// sample for each image leaf and selecting the sampled result stays in portable Logical SPIR-V. The
/// pass declines anything except an image-only select tree whose complete consumer closure is
/// explicit-LOD sampling.
pub(crate) fn rewrite_opaque_image_selects_module(module: &mut Module) -> Result<(), String> {
    if !opaque_image_select::rewrite_opaque_image_selects(module) {
        return Err("native emitter: no lowerable opaque image select".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Remodel every Workgroup variable accessed only as the float-as-int atomic idiom (the
/// `OpBitcast %_ptr_Workgroup_<int> %chain` → `OpAtomicSMin/SMax` pattern that spirv-val rejects as an
/// illegal logical-pointer bitcast) so its float leaves become the int the atomics use. Errors if no
/// variable was remodeled. Byte-safe by construction (Workgroup scratch, float↔int32 bit-identical,
/// layout-preserving clone, strict all-uses gate). The caller adopts the result ONLY if it
/// independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_workgroup_atomic_floats_module(module: &mut Module) -> Result<(), String> {
    if !wg_atomic::rewrite_workgroup_atomic_floats(module) {
        return Err("native emitter: no workgroup float-as-int atomic to remodel".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Retype every Function scalar-integer variable accessed ONLY as the sub-word packed-scalar idiom
/// (e.g. a `uint` alloca written as two `half` lanes then read whole) into a `<N x E>` vector, in
/// place. This drops the illegal scalar-indexing access chains' invalidity and value-bitcasts its
/// whole-word loads/stores. Errors if no variable was remodeled. Byte-safe by construction
/// (Function scratch, little-endian-identical vector layout). The caller adopts the result ONLY if
/// it independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_subword_packed_scalars_module(module: &mut Module) -> Result<(), String> {
    if !subword_pack::rewrite_subword_packed_scalars(module) {
        return Err("native emitter: no sub-word packed scalar to remodel".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply the M4 phi-the-index retry to an already-loaded module and re-finalize it in place. Errors
/// if no eligible illegal logical-pointer `OpPhi` was rewritten. The caller adopts the result ONLY
/// if it independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_logical_pointer_phis_retry_module(module: &mut Module) -> Result<(), String> {
    if !phi_index::rewrite_logical_pointer_phis(module) {
        return Err("native emitter: no logical pointer phi to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply the M4 phi-the-index legalization to an in-flight module (the PRIMARY emit tail), in place.
/// Rewrites only ILLEGAL logical-pointer phis — a non-`StorageBuffer`/`Workgroup` (Private/Function/
/// UniformConstant) pointer `OpPhi`, which is always spirv-val-invalid — so it can only move an
/// already-invalid module toward valid, never touch a validating one (floor-safe by construction).
/// Same mechanism the retry tier ([`rewrite_logical_pointer_phis_retry_module`]) applies, hoisted
/// onto the no-retry path so these functions' PRIMARY emit validates instead of shipping via retry
/// rescue. Returns true if any phi was rewritten. The caller runs `canonicalize_ids` afterward.
pub(crate) fn rewrite_logical_pointer_phis_module(module: &mut Module) -> bool {
    phi_index::rewrite_logical_pointer_phis(module)
}

/// Legalize integer `OpPhi` result/incoming width mismatches in an in-flight module (the PRIMARY emit
/// tail) by truncating a wide incoming to the phi's narrower integer result type. Only touches phis
/// that are already spirv-val-INVALID (an integer phi whose operand type differs from its result
/// type), so it can only move an already-invalid module toward valid — floor-safe by construction.
/// See [`phi_index::rewrite_integer_width_phis`] for the mechanism. The caller runs `canonicalize_ids`
/// afterward. Returns true if any operand was converted.
pub(crate) fn rewrite_integer_width_phis_module(module: &mut Module) -> bool {
    phi_index::rewrite_integer_width_phis(module)
}

/// Register-demote any value in an in-flight module (the PRIMARY emit tail) whose defining block no
/// longer dominates a use — the loop-closed-SSA violation the `MultipleExits` funnel
/// (`synth_multi_exit_merge`) leaves behind (spirv-val: *"ID X defined in block B does not dominate its
/// use in block U"*). Spills the value to a function-scope `OpVariable` (stored after its def, loaded
/// before each non-dominated use). Only touches modules that ALREADY carry such a violation (a valid
/// module has every def dominating its uses), so it is floor-safe by construction. See
/// [`cfg::demote_nondominating_values`]. The caller runs `canonicalize_ids` afterward. Returns true if
/// any value was demoted.
pub(crate) fn demote_nondominating_values_module(module: &mut Module) -> bool {
    cfg::demote_nondominating_values(module)
}

/// Node-split a MULTI-ENTRY loop in an in-flight module (the PRIMARY emit tail) whose header is entered
/// by forward edges from two different selections' arms — the irreducible shape `structured_plan`
/// over-admits, spirv-val-INVALID (*"block X exits the selection headed by Y, but not via a structured
/// exit"*; the mlx-steel `steel_attention` family). Duplicates the loop region for the inner arm's
/// entry, routing the clone's exit to that selection's merge, so each loop is single-entry. Only fires
/// on a loop with ≥2 forward header entries (a valid loop is single-entry), so it is floor-safe by
/// construction. See [`cfg::split_multientry_loop_selection_exits`]. The caller runs `canonicalize_ids`
/// afterward. Returns true if any loop was split.
pub(crate) fn split_multientry_loop_selection_exits_module(module: &mut Module) -> bool {
    cfg::split_multientry_loop_selection_exits(module)
}

/// Lower a cross-binding pointer `OpSelect`/`OpPhi` (pointers into DISTINCT buffer bindings, spirv-val-
/// INVALID *"Variable pointers must point into the same structure"*) INTO THE VALUE DOMAIN in an
/// in-flight module (the PRIMARY emit tail): load from every candidate buffer, select among the LOADED
/// VALUES. This is the SAME portable value-domain form the `value_select` retry tier ships
/// (`rewrite_cross_binding_pointer_merges_to_values_module`); running it on the primary path makes the
/// direct emit valid instead of relying on retry-rescue. The caller runs `canonicalize_ids` afterward.
/// Returns true if a merge was value-lowered.
///
/// GUARDED to `StorageBuffer`-pointer merges — the genuine "Variable pointers must point into the same
/// structure" class the `value_select` retry rescues. A merge over LOGICAL (`Private`/`Function`/
/// `Workgroup`) pointers is a DIFFERENT population: the unmodeled-device-buffer placeholder family
/// (`emit_private_zero_pointer_value`, e.g. `01/a70fb990` `%_ptr_Private_float %86` dynamically indexed
/// → the "reached non-composite" error), which the PRIMARY error routes to `fc_promote_psb` /
/// pointer-typing retry, NOT `value_select`. Value-lowering such a Private merge does NOT resolve the
/// module (the non-composite issue remains) and DERAILS that pointer-typing rescue (`a70fb990` regressed
/// valid→FALLBACK), so it is excluded — mirroring production, where `value_select` never fires for a
/// non-`CrossBindingPointerMerge` error class. Decides purely from IR pointer storage class, never a
/// shader name.
pub(crate) fn value_lower_cross_binding_pointer_merges_module(module: &mut Module) -> bool {
    let defs = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
        .collect::<HashMap<_, _>>();
    let has_storage_buffer_pointer_merge = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| matches!(inst.class.opcode, Op::Phi | Op::Select))
        .filter_map(|inst| inst.result_type)
        .any(|ty| ptr_storage(&defs, ty) == Some(StorageClass::StorageBuffer));
    if !has_storage_buffer_pointer_merge {
        return false;
    }
    psb_value_select::rewrite_cross_binding_pointer_merges_to_values(module)
}

/// Apply the phi-the-index rewrite in place to VARIABLE-pointer (`StorageBuffer`/`Workgroup`) phis.
/// These phis are legal SPIR-V under
/// `VariablePointersStorageBuffer` — spirv-val passes — but MoltenVK's SPIRV-Cross MSL backend
/// cannot always express them (pipeline creation fails with `cannot initialize a variable of type
/// 'device float *' with an lvalue of type 'device float'`). The index-phi form is semantically
/// identical (same base, same per-arm indices, one rematerialized access chain), so the caller runs
/// this as a PORTABILITY NORMALIZATION on the success path and adopts the result only if it
/// independently validates. Errors if no eligible phi was rewritten.
pub(crate) fn rewrite_variable_pointer_phis_module(module: &mut Module) -> Result<(), String> {
    if !phi_index::rewrite_variable_pointer_phis(module) {
        return Err("native emitter: no variable pointer phi to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    crate::passes::lower_scalar_i64_arithmetic_module(module);
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply static constant-branch pruning (function-constant dead-arm DCE) in place. Errors if
/// nothing was pruned. The caller (the failure-triggered retry) adopts the result ONLY if it
/// independently validates, and the transformation removes only statically-unreachable code +
/// unused pure values, so it is floor-safe AND semantics-preserving by construction.
pub(crate) fn prune_constant_branches_module(module: &mut Module) -> Result<(), String> {
    if !constfold::prune_constant_branches(module) {
        return Err("native emitter: no constant branch to prune".to_string());
    }
    drop_dangling_debug_targets_module(module);
    add_native_module_capabilities(module);
    Ok(())
}

/// Preserving form of [`prune_constant_branches_module`] for a primary module that still carries
/// typed sidecar roots. Returns whether pruning changed the module.
pub(crate) fn prune_constant_branches_module_preserving(
    module: &mut Module,
    preserved_global_ids: &[Word],
) -> bool {
    let roots = preserved_global_ids.iter().copied().collect();
    let changed = constfold::prune_constant_branches_preserving(module, &roots);
    let dropped = drop_dangling_debug_targets_module(module);
    changed || dropped
}

/// Whether the module contains an `OpFunctionCall` to a BODILESS `llvm.agx*` hardware-intrinsic
/// declaration (AGX matmul `igemm`, `load/store.with.emask`, …) — the structural trigger for the
/// primary-path FC prune in `primary_retry.rs`. Such a call is never executable on a Vulkan target:
/// no lowering exists and the declaration has no body. Keyed on the `llvm.agx` ABI-symbol namespace
/// via the `OpName` of bodiless functions (the emitter always names emitted declarations) — never a
/// shader name.
pub(crate) fn has_bodiless_agx_call_module(module: &Module) -> bool {
    use std::collections::HashSet;
    let agx_names: HashSet<spirv::Word> = module
        .debug_names
        .iter()
        .filter(|inst| inst.class.opcode == spirv::Op::Name)
        .filter_map(|inst| {
            let Operand::IdRef(id) = inst.operands.first()? else {
                return None;
            };
            let Operand::LiteralString(name) = inst.operands.get(1)? else {
                return None;
            };
            name.starts_with("llvm.agx").then_some(*id)
        })
        .collect();
    if agx_names.is_empty() {
        return false;
    }
    let bodiless: HashSet<spirv::Word> = module
        .functions
        .iter()
        .filter(|f| f.blocks.is_empty())
        .filter_map(|f| f.def.as_ref().and_then(|d| d.result_id))
        .filter(|id| agx_names.contains(id))
        .collect();
    if bodiless.is_empty() {
        return false;
    }
    module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == spirv::Op::FunctionCall
                    && matches!(
                        inst.operands.first(),
                        Some(Operand::IdRef(callee)) if bodiless.contains(callee)
                    )
            })
        })
    })
}

/// Reconcile whole-buffer scalar fallback arms in place (byte-0 base of an FC-multiplexed
/// raw-modeled device buffer) to the merge's scalar pointee, so a cross-binding pointer merge's arms
/// share one pointee type (see [`buffer_arm_reconcile`]). Errors if nothing was reconciled. Applied
/// only in the adopt-if-VALIDATES `fc_promote_psb` retry (feeding PSB), so floor-safe by
/// construction; byte-EXACT (element 0 = offset 0 for either scalar element type).
pub(crate) fn reconcile_whole_buffer_scalar_arms_module(module: &mut Module) -> Result<(), String> {
    if !buffer_arm_reconcile::reconcile_whole_buffer_scalar_arms(module) {
        return Err("native emitter: no whole-buffer scalar arm to reconcile".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply the W2 relooper (switch-dispatch + register demotion) in place with the default block cap.
/// Errors if no function was rewritten. The caller (the failure-triggered cfg retry) adopts the
/// result ONLY if it independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_to_relooper_module(module: &mut Module) -> Result<(), String> {
    rewrite_to_relooper_module_capped(module, relooper::default_max_relooper_blocks())
}

/// Higher block budget for the function-constant-dead prune → relooper composition. Pruning can
/// shrink an otherwise oversized function below this ceiling while it remains above the normal
/// relooper budget.
pub const PRUNE_THEN_RELOOPER_MAX_BLOCKS: usize = 2048;

/// Bounded high cap for the relooper retry over an unrepaired CFG that the structurizer emitted as a
/// REJECT (complete but spirv-val-invalid bytes). The normal relooper stays capped at 1024 blocks;
/// this retry-only cap is sufficient for the complete graph a reject function emits, while remaining
/// below the 8192 diagnostic ceiling for the separately measured huge emitted-module path.
pub const CFG_EMIT_RELOOPER_MAX_BLOCKS: usize = 2048;

/// Like [`rewrite_to_relooper_module`] but with an explicit block cap. The prune-then-relooper
/// retry uses a higher cap than the default 1024: its input is a function whose statically-dead
/// function-constant arms were already pruned away (byte-correct DCE), so a >1024-block source can
/// land below the cap after pruning yet still exceed the default. Lifting the cap only for that
/// already-failing, adopt-if-validates path keeps the normal relooper's perf budget intact while
/// admitting the pruned huge-function cfg cases.
pub(crate) fn rewrite_to_relooper_module_capped(
    module: &mut Module,
    max_blocks: usize,
) -> Result<(), String> {
    if !relooper::rewrite_to_relooper(module, max_blocks) {
        return Err("native emitter: no function to relooper".to_string());
    }
    drop_dangling_debug_targets_module(module);
    add_native_module_capabilities(module);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function};
    use spirv::Decoration;

    fn inst(op: Op, ty: Option<Word>, id: Option<Word>, operands: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, id, operands)
    }

    fn workgroup_struct_byte_store_module(byte_offset: u32) -> Module {
        let mut module = Module::default();
        module.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeInt,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeStruct,
                None,
                Some(3),
                vec![Operand::IdRef(2), Operand::IdRef(1)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(3),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(1),
                ],
            ),
            inst(
                Op::Constant,
                Some(2),
                Some(6),
                vec![Operand::LiteralBit32(byte_offset)],
            ),
            inst(Op::ConstantNull, Some(1), Some(7), vec![]),
            inst(
                Op::Variable,
                Some(4),
                Some(8),
                vec![Operand::StorageClass(StorageClass::Workgroup)],
            ),
        ];
        module.annotations = vec![
            inst(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(3),
                    Operand::LiteralBit32(0),
                    Operand::Decoration(Decoration::Offset),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(3),
                    Operand::LiteralBit32(1),
                    Operand::Decoration(Decoration::Offset),
                    Operand::LiteralBit32(4),
                ],
            ),
        ];
        module.functions = vec![Function {
            blocks: vec![Block {
                label: None,
                instructions: vec![
                    inst(
                        Op::PtrAccessChain,
                        Some(5),
                        Some(9),
                        vec![Operand::IdRef(8), Operand::IdRef(6)],
                    ),
                    inst(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(9), Operand::IdRef(7)],
                    ),
                ],
            }],
            ..Default::default()
        }];
        module
    }

    #[test]
    fn drops_workgroup_struct_padding_byte_zero_store() {
        let mut module = workgroup_struct_byte_store_module(5);

        assert!(drop_workgroup_struct_padding_byte_zero_stores_module(
            &mut module
        ));
        let insts = &module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(9)),
            "padding byte chain should be removed"
        );
        assert!(
            !insts.iter().any(|inst| {
                inst.class.opcode == Op::Store
                    && matches!(inst.operands.first(), Some(Operand::IdRef(9)))
            }),
            "padding zero store should be removed"
        );
    }

    #[test]
    fn drops_derived_workgroup_struct_padding_byte_zero_stores() {
        let mut module = workgroup_struct_byte_store_module(6);
        module.types_global_values.extend([
            inst(
                Op::Constant,
                Some(2),
                Some(10),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(
                Op::TypeInt,
                None,
                Some(12),
                vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
            ),
            inst(Op::ConstantNull, Some(12), Some(13), vec![]),
            inst(
                Op::Constant,
                Some(2),
                Some(16),
                vec![Operand::LiteralBit32(8)],
            ),
        ]);
        module.functions[0].blocks[0].instructions = vec![
            inst(
                Op::PtrAccessChain,
                Some(5),
                Some(9),
                vec![Operand::IdRef(8), Operand::IdRef(6)],
            ),
            inst(Op::UConvert, Some(1), Some(14), vec![Operand::IdRef(13)]),
            inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(9), Operand::IdRef(14)],
            ),
            inst(
                Op::PtrAccessChain,
                Some(5),
                Some(11),
                vec![Operand::IdRef(9), Operand::IdRef(10)],
            ),
            inst(
                Op::ShiftRightLogical,
                Some(12),
                Some(15),
                vec![Operand::IdRef(13), Operand::IdRef(16)],
            ),
            inst(Op::UConvert, Some(1), Some(17), vec![Operand::IdRef(15)]),
            inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(11), Operand::IdRef(17)],
            ),
        ];

        assert!(drop_workgroup_struct_padding_byte_zero_stores_module(
            &mut module
        ));
        let insts = &module.functions[0].blocks[0].instructions;
        assert!(
            !insts
                .iter()
                .any(|inst| inst.result_id == Some(9) || inst.result_id == Some(11)),
            "direct and derived padding byte chains should be removed"
        );
        assert!(
            !insts.iter().any(|inst| inst.class.opcode == Op::Store),
            "all zero padding byte stores should be removed"
        );
    }

    #[test]
    fn keeps_workgroup_struct_member_byte_zero_store() {
        let mut module = workgroup_struct_byte_store_module(4);

        assert!(!drop_workgroup_struct_padding_byte_zero_stores_module(
            &mut module
        ));
        let insts = &module.functions[0].blocks[0].instructions;
        assert!(insts.iter().any(|inst| inst.result_id == Some(9)));
    }

    #[test]
    fn drops_debug_records_for_undefined_targets() {
        let mut module = Module::default();
        module.types_global_values = vec![inst(
            Op::TypeInt,
            None,
            Some(1),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        )];
        module.debug_names = vec![
            inst(
                Op::Name,
                None,
                None,
                vec![Operand::IdRef(1), Operand::LiteralString("live".into())],
            ),
            inst(
                Op::Name,
                None,
                None,
                vec![Operand::IdRef(99), Operand::LiteralString("dead".into())],
            ),
        ];
        module.annotations = vec![
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(1),
                    Operand::Decoration(Decoration::RelaxedPrecision),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(99),
                    Operand::Decoration(Decoration::RelaxedPrecision),
                ],
            ),
        ];

        assert!(drop_dangling_debug_targets_module(&mut module));
        assert_eq!(module.debug_names.len(), 1);
        assert_eq!(module.annotations.len(), 1);
        assert_eq!(
            module.debug_names[0].operands.first(),
            Some(&Operand::IdRef(1))
        );
        assert_eq!(
            module.annotations[0].operands.first(),
            Some(&Operand::IdRef(1))
        );
    }
}
