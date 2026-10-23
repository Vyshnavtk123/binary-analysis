# Crypto accuracy fixtures

The optional end-to-end test builds these Go sources as Windows x64 executables and scans them with the normal PE pipeline:

| Fixture | Expected evidence |
| --- | --- |
| `aes` | AES function symbol and call-site evidence |
| `rsa` | RSA function symbol and call-site evidence |
| `string_only` | AES is `string_only`; no symbol or call claim |
| `unreachable` | Dead AES code is not reported as a linked AES function |
| `indirect` | A resolved indirect edge or an explicit unresolved indirect-call record |

Run `py -m pip install -r requirements-dev.txt`, install a Go toolchain, then run `py -m pytest -q tests/test_crypto_analysis_evidence.py tests/test_crypto_go_binary_fixtures.py`.

The binary tests are skipped when Go, `pefile`, or Capstone is unavailable. They do not execute the generated sample binaries.
