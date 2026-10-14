# Native Windows product gate, 2026-10-14

The four product crates (`metal2vulkan`, `metal2vulkan-air`,
`metal2vulkan-runtime`, `metal2vulkan-devtools`) passed all 4,543 selected
tests with all features under nextest. The 22 existing ignored tests remain
ignored. Formatting, all-target/all-feature clippy with warnings denied and
the product rustdoc checks passed. Native Vulkan coherence and private-memory
checks separately passed with zero validation-layer errors.

This was a resumed gate, not one uninterrupted run. Exact successful
`(binary_id, test_name)` pairs were retained: 2,192 initial passes, 173 in
the first continuation, 129 in the next, one corrected scratch-path case,
and 2,048 in the final continuation. The union contains 4,543 unique passes;
failed tests were never excluded as successes. No GPU test ran alongside a
VM boot. The validation/corpus executor crate is not one of these four
product crates and is not claimed to have a Windows workspace-wide gate.

Three fixture problems were corrected without relaxing comparisons:

- Native source and text fixtures must use LF bytes. The committed
  `.gitattributes` makes this independent of the Windows checkout setting.
- LLVM IR decimal operands must exactly represent the intended float.
  Non-exact decimal f32 operands in the authored gather fixture now use
  LLVM's canonical hexadecimal representation of those same f32 values.
- Diagnostic labels can contain punctuation forbidden in a Windows path.
  Scratch helpers retain their existing PID/counter uniqueness and keep
  arbitrary labels in error messages rather than directory names.

The last two fixes change only test modules. The production translator,
runtime and their source fingerprint are unchanged by them. The calibrated
native Windows profile still advertises only the measured conservative
16 KiB function-storage floor for the exact GPU/driver documented in the
runtime; this test cleanup does not raise it.
