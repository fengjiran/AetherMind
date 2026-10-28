# Frozen final-validation policy

This directory pins the reviewed test populations for the integrated five-batch
matrix. The policy is test metadata, not a numerical oracle or a capability
registry. Runtime/model correctness still comes from the actual tests.

For each CTest slice, expected_count and names_sha256 describe the complete
filtered `ctest --show-only=json-v1` inventory, including automatically added
fixture setup tests. Do not compute it by filtering the labels of an unfiltered
inventory: that can omit fixtures which CTest will actually run.

The CLI and model-benchmark entries similarly pin all exact verbose unittest
case IDs, for example `test_name (__main__.ClassName.test_name)`.

The digest is SHA256 of the sorted JSON string list encoded as UTF8, using
`ensure_ascii=False` and compact separators `(",", ":")`. Input names must be
unique; required runs reject empty populations, disabled cases, skips, errors,
missing/extra names and non-finite elapsed times.

Run the documented [final validation entrypoint](../../../docs/guides/final-validation.md).
To review an intended change, retain the old policy, generate a candidate
inventory, and inspect the exact name differences:

```bash
python3 tools/verify_ctest_report.py \
  --inventory <slice>.inventory.json --inventory-only
```

The checker only prints a candidate count/digest. No normal build, fixture tool,
CI job or validation runner rewrites this file. Changes require an explicit
review of the corresponding source/filter/fixture change. A count match alone
does not authorize replacing the expected digest.
