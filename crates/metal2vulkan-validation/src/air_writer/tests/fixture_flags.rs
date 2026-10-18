use std::ffi::OsString;

pub(crate) fn compile_flags(text: &str) -> Vec<OsString> {
    [
        ("// metal-std:", "-std="),
        ("// metal-macos-min:", "-mmacosx-version-min="),
    ]
    .into_iter()
    .filter_map(|(key, flag)| {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(key))
            .map(|value| format!("{flag}{}", value.trim()).into())
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::compile_flags;
    use std::ffi::OsString;

    #[test]
    fn fixture_standard_and_minimum_target_are_explicit_arguments() {
        assert_eq!(
            compile_flags("// metal-std: metal4.1\n// metal-macos-min: 27.0\n"),
            ["-std=metal4.1", "-mmacosx-version-min=27.0"].map(OsString::from)
        );
        assert_eq!(
            compile_flags("// metal-std: metal4.0\n"),
            [OsString::from("-std=metal4.0")]
        );
        assert_eq!(
            compile_flags("// metal-macos-min: 26.2\n"),
            [OsString::from("-mmacosx-version-min=26.2")]
        );
        assert!(compile_flags("// No fixture overrides.\n").is_empty());
    }
}
