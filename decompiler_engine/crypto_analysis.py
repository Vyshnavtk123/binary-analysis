from __future__ import annotations

"""Bounded cryptography indicator extraction from PE strings and imports.

This pass reports textual/API evidence only. It does not prove that an algorithm
is executed or infer algorithms from arbitrary machine-code constants.
"""

import re
import hashlib
import json
import uuid
from pathlib import Path
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Iterable

ANALYZER_NAME = "Independent Decompiler Engine"
ANALYZER_VERSION = "0.4.0-dev"


# Canonical names with common spellings, provider labels, and well-known aliases.
# This is a practical catalog, not an exhaustive registry of every algorithm.
_ALGORITHMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("RSA", "public_key", ("RSA", "RSA key", "RSA encryption")),
    ("RSA-PSS", "signature", ("RSA-PSS", "RSASSA-PSS")),
    ("RSA-OAEP", "key_encapsulation", ("RSA-OAEP", "RSAES-OAEP")),
    ("DSA", "signature", ("DSA", "Digital Signature Algorithm")),
    ("Diffie-Hellman", "key_exchange", ("Diffie-Hellman", "Diffie Hellman")),
    ("DHE", "key_exchange", ("DHE", "DH key exchange")),
    ("ECDH", "key_exchange", ("ECDH", "Elliptic Curve Diffie-Hellman")),
    ("ECDHE", "key_exchange", ("ECDHE", "ECDHE key exchange")),
    ("ECDSA", "signature", ("ECDSA",)),
    ("Ed25519", "signature", ("Ed25519", "Edwards25519")),
    ("Ed448", "signature", ("Ed448", "Edwards448")),
    ("X25519", "key_exchange", ("X25519", "Curve25519")),
    ("X448", "key_exchange", ("X448", "Curve448")),
    ("ML-KEM", "post_quantum_kem", ("ML-KEM", "MLKEM", "FIPS 203")),
    ("ML-KEM-512", "post_quantum_kem", ("ML-KEM-512", "MLKEM512", "Kyber512")),
    ("ML-KEM-768", "post_quantum_kem", ("ML-KEM-768", "MLKEM768", "Kyber768")),
    ("ML-KEM-1024", "post_quantum_kem", ("ML-KEM-1024", "MLKEM1024", "Kyber1024")),
    ("Kyber", "post_quantum_kem", ("Kyber", "CRYSTALS-Kyber")),
    ("ML-DSA", "post_quantum_signature", ("ML-DSA", "MLDSA", "FIPS 204")),
    ("ML-DSA-44", "post_quantum_signature", ("ML-DSA-44", "MLDSA44", "Dilithium2")),
    ("ML-DSA-65", "post_quantum_signature", ("ML-DSA-65", "MLDSA65", "Dilithium3")),
    ("ML-DSA-87", "post_quantum_signature", ("ML-DSA-87", "MLDSA87", "Dilithium5")),
    ("SLH-DSA", "post_quantum_signature", ("SLH-DSA", "SLHDSA", "SPHINCS+", "SPHINCSPLUS", "FIPS 205")),
    ("FN-DSA", "post_quantum_signature", ("FN-DSA", "FNDSA", "Falcon-512", "Falcon-1024")),
    ("XMSS", "post_quantum_signature", ("XMSS", "eXtended Merkle Signature Scheme")),
    ("LMS", "post_quantum_signature", ("LMS", "Leighton-Micali Signature")),
    ("Hybrid-KEM", "hybrid_key_encapsulation", ("Hybrid-KEM", "Hybrid KEM", "ML-KEM-X25519", "X25519MLKEM")),
    ("Hybrid-TLS", "hybrid_protocol", ("Hybrid-TLS", "Hybrid TLS", "post-quantum TLS")),
    ("AES", "symmetric_cipher", ("AES", "Rijndael")),
    ("AES-GCM", "authenticated_cipher", ("AES-GCM", "AES GCM")),
    ("AES-CCM", "authenticated_cipher", ("AES-CCM", "AES CCM")),
    ("AES-CBC", "block_cipher_mode", ("AES-CBC", "AES CBC")),
    ("AES-CTR", "block_cipher_mode", ("AES-CTR", "AES CTR")),
    ("AES-XTS", "block_cipher_mode", ("AES-XTS", "AES XTS")),
    ("AES-ECB", "block_cipher_mode", ("AES-ECB", "AES ECB")),
    ("ChaCha20", "symmetric_cipher", ("ChaCha20", "ChaCha")),
    ("ChaCha20-Poly1305", "authenticated_cipher", ("ChaCha20-Poly1305", "ChaCha20 Poly1305")),
    ("Poly1305", "message_authentication", ("Poly1305",)),
    ("XChaCha20", "symmetric_cipher", ("XChaCha20", "XChaCha20-Poly1305")),
    ("3DES", "symmetric_cipher", ("3DES", "Triple DES", "TDEA")),
    ("RC4", "stream_cipher", ("RC4", "ARCFOUR")),
    ("DES", "symmetric_cipher", ("DES", "Data Encryption Standard")),
    ("Camellia", "symmetric_cipher", ("Camellia",)),
    ("Twofish", "symmetric_cipher", ("Twofish",)),
    ("Blowfish", "symmetric_cipher", ("Blowfish",)),
    ("SM4", "symmetric_cipher", ("SM4",)),
    ("SHA-1", "hash", ("SHA-1", "SHA1")),
    ("SHA-2", "hash_family", ("SHA-2", "SHA2")),
    ("SHA-224", "hash", ("SHA-224", "SHA224")),
    ("SHA-256", "hash", ("SHA-256", "SHA256")),
    ("SHA-384", "hash", ("SHA-384", "SHA384")),
    ("SHA-512", "hash", ("SHA-512", "SHA512")),
    ("SHA-3", "hash_family", ("SHA-3", "SHA3")),
    ("SHA3-256", "hash", ("SHA3-256", "SHA3_256")),
    ("SHA3-384", "hash", ("SHA3-384", "SHA3_384")),
    ("SHA3-512", "hash", ("SHA3-512", "SHA3_512")),
    ("SHAKE128", "extendable_output_hash", ("SHAKE128", "SHAKE-128")),
    ("SHAKE256", "extendable_output_hash", ("SHAKE256", "SHAKE-256")),
    ("MD5", "hash", ("MD5",)),
    ("BLAKE2", "hash_family", ("BLAKE2", "BLAKE2b", "BLAKE2s")),
    ("BLAKE3", "hash", ("BLAKE3",)),
    ("Keccak", "hash_family", ("Keccak",)),
    ("HMAC", "message_authentication", ("HMAC",)),
    ("HMAC-SHA1", "message_authentication", ("HMAC-SHA1", "HMAC-SHA-1")),
    ("HMAC-SHA2", "message_authentication", ("HMAC-SHA2", "HMAC-SHA-2")),
    ("HMAC-SHA3", "message_authentication", ("HMAC-SHA3", "HMAC-SHA-3")),
    ("HMAC-MD5", "message_authentication", ("HMAC-MD5",)),
    ("GMAC", "message_authentication", ("GMAC",)),
    ("CMAC", "message_authentication", ("CMAC",)),
    ("KMAC", "message_authentication", ("KMAC",)),
    ("HKDF", "key_derivation", ("HKDF",)),
    ("PBKDF2", "key_derivation", ("PBKDF2", "PBKDF2-HMAC")),
    ("Argon2", "password_hash", ("Argon2", "Argon2id", "Argon2i", "Argon2d")),
    ("scrypt", "password_hash", ("scrypt",)),
    ("TLS", "protocol", ("TLS", "Transport Layer Security")),
    ("Mutual TLS", "protocol", ("Mutual TLS", "mTLS")),
    ("DTLS", "protocol", ("DTLS", "Datagram TLS")),
    ("SSL", "protocol", ("SSL", "Secure Sockets Layer")),
    ("X.509", "certificate_format", ("X.509", "X509")),
    ("PKCS", "standard", ("PKCS#1", "PKCS1", "PKCS#7", "PKCS7", "PKCS#8", "PKCS8", "PKCS#12", "PKCS12")),
)

_KEYWORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cryptography", "crypto_concept", ("cryptography", "cryptographic", "crypto")),
    ("encryption", "crypto_operation", ("encrypt", "encryption", "encrypted")),
    ("decryption", "crypto_operation", ("decrypt", "decryption", "decrypted")),
    ("cipher", "crypto_concept", ("cipher", "ciphertext", "ciphersuite")),
    ("key exchange", "key_management", ("key exchange", "key_exchange", "keyexchange")),
    ("public key", "key_management", ("public key", "public_key", "publickey")),
    ("private key", "key_management", ("private key", "private_key", "privatekey")),
    ("key pair", "key_management", ("key pair", "keypair", "key_pair")),
    ("certificate", "pki", ("certificate", "certificates", "x509")),
    ("handshake", "protocol", ("handshake",)),
    ("digital signature", "signature", ("digital signature", "signature", "signing")),
    ("digest", "hashing", ("digest", "message digest")),
    ("random generator", "randomness", ("random number generator", "cryptgenrandom", "bcryptgenrandom", "cng system rng")),
    ("nonce", "crypto_parameter", ("nonce",)),
    ("salt", "crypto_parameter", ("salt",)),
    ("initialization vector", "crypto_parameter", ("initialization vector", "init vector", "iv")),
    ("key derivation", "key_management", ("key derivation", "key derivation function", "kdf")),
    ("post-quantum", "post_quantum", ("post-quantum", "postquantum", "quantum-safe", "quantum resistant")),
    ("FIPS", "standard", ("FIPS", "Federal Information Processing Standards")),
    ("OCSP", "pki", ("OCSP",)),
    ("PKI", "pki", ("PKI", "public key infrastructure")),
)

_CRYPTO_IMPORT_MARKERS = (
    "bcrypt", "ncrypt", "crypt32", "advapi32!crypt", "cryptsp", "cryptbase",
    "libcrypto", "libssl", "openssl", "gnutls", "mbedtls", "wolfssl",
    "libsodium", "botan", "nettle", "schannel", "secur32!initializeSecurityContext",
    "secur32!acquireCredentialsHandle", "cryptencrypt", "cryptdecrypt",
    "cryptgenkey", "cryptderivekey", "cryptacquirecontext", "cryptimportkey",
)

_CRYPTO_MODULE_MARKERS = (
    "golang.org/x/crypto", "github.com/cloudflare/circl", "github.com/gtank/ristretto",
    "github.com/consensys/gnark-crypto", "github.com/decred/dcrd/dcrec",
)

# Package paths are stronger binary evidence than standalone algorithm words:
# they usually come from retained language/runtime metadata or linked package
# symbols. They still do not prove that a particular function executes.
_CRYPTO_PACKAGE_PATHS: tuple[tuple[str, str], ...] = (
    ("crypto/aes", "AES"), ("crypto/des", "DES"), ("crypto/rc4", "RC4"),
    ("crypto/rsa", "RSA"), ("crypto/dsa", "DSA"), ("crypto/ecdsa", "ECDSA"),
    ("crypto/ecdh", "ECDH"), ("crypto/elliptic", "ECC"),
    ("crypto/ed25519", "Ed25519"), ("crypto/md5", "MD5"),
    ("crypto/sha1", "SHA-1"), ("crypto/sha256", "SHA-256"),
    ("crypto/sha512", "SHA-512"), ("crypto/hmac", "HMAC"),
    ("crypto/cipher", "cipher API"), ("crypto/tls", "TLS"),
    ("crypto/x509", "X.509"), ("golang.org/x/crypto/chacha20poly1305", "ChaCha20-Poly1305"),
    ("golang.org/x/crypto/curve25519", "X25519"),
    ("golang.org/x/crypto/sha3", "SHA-3"),
    ("golang.org/x/crypto/argon2", "Argon2"),
    ("golang.org/x/crypto/pbkdf2", "PBKDF2"),
    ("golang.org/x/crypto/hkdf", "HKDF"), ("golang.org/x/crypto/chacha20", "ChaCha20"),
    ("golang.org/x/crypto/blake2b", "BLAKE2"),
    ("golang.org/x/crypto/blake2s", "BLAKE2"),
    ("golang.org/x/crypto/bcrypt", "bcrypt"), ("golang.org/x/crypto/scrypt", "scrypt"),
    ("golang.org/x/crypto/blowfish", "Blowfish"), ("golang.org/x/crypto/twofish", "Twofish"),
    ("github.com/cloudflare/circl/kem/mlkem", "ML-KEM"),
    ("github.com/cloudflare/circl/sign/mldsa", "ML-DSA"),
    ("github.com/cloudflare/circl/sign/slhdsa", "SLH-DSA"),
    ("crypto/mlkem", "ML-KEM"), ("crypto/sha3", "SHA-3"),
)


@dataclass(frozen=True)
class CryptoSignatureRule:
    """Named Go symbol rule; string aliases never satisfy symbol requirements."""
    symbol_pattern: str
    algorithm: str | None
    family: str | None
    operation: str
    variant: str | None = None
    aliases: tuple[str, ...] = ()
    evidence_requirements: tuple[str, ...] = ("go_function_symbol",)


CRYPTO_SIGNATURE_RULES: tuple[CryptoSignatureRule, ...] = (
    CryptoSignatureRule(r"crypto/aes\.NewCipher", "AES", "AES", "cipher_initialization", aliases=("AES", "Rijndael")),
    CryptoSignatureRule(r"crypto/cipher\.NewGCM", None, None, "aead_mode_gcm", aliases=("GCM",), evidence_requirements=("go_function_symbol", "underlying_cipher_identity")),
    CryptoSignatureRule(r"crypto/cipher\.NewGCMWithNonceSize", None, None, "aead_mode_gcm_custom_nonce", aliases=("GCM",), evidence_requirements=("go_function_symbol", "underlying_cipher_identity")),
    CryptoSignatureRule(r"crypto/cipher\.NewCBC(Encrypter|Decrypter)", None, None, "block_mode_cbc", aliases=("CBC",), evidence_requirements=("go_function_symbol", "underlying_cipher_identity")),
    CryptoSignatureRule(r"crypto/cipher\.NewCTR", None, None, "block_mode_ctr", aliases=("CTR",), evidence_requirements=("go_function_symbol", "underlying_cipher_identity")),
    CryptoSignatureRule(r"crypto/rsa\.(GenerateKey|GenerateMultiPrimeKey)", "RSA", "RSA", "key_generation", aliases=("RSA",)),
    CryptoSignatureRule(r"crypto/rsa\.(SignPSS|VerifyPSS)", "RSA", "RSA", "signature_pss", "PSS", ("RSA-PSS",)),
    CryptoSignatureRule(r"crypto/rsa\.(SignPKCS1v15|VerifyPKCS1v15)", "RSA", "RSA", "signature_pkcs1v15", "PKCS#1 v1.5", ("RSA",)),
    CryptoSignatureRule(r"crypto/rsa\.(EncryptOAEP|DecryptOAEP)", "RSA", "RSA", "encryption_oaep", "OAEP", ("RSA-OAEP",)),
    CryptoSignatureRule(r"crypto/ecdsa\.(Sign|Verify|SignASN1|VerifyASN1)", "ECDSA", "ECDSA", "signature", aliases=("ECDSA",)),
    CryptoSignatureRule(r"crypto/ecdh\..*", "ECDH", "ECDH", "key_agreement", aliases=("ECDH",)),
    CryptoSignatureRule(r"crypto/sha256\.(New|Sum256)", "SHA-256", "SHA-2", "hash", "SHA-256", ("SHA-256", "SHA256")),
    CryptoSignatureRule(r"crypto/sha512\.(New|Sum512|Sum384)", "SHA-512", "SHA-2", "hash", "SHA-512", ("SHA-512", "SHA512")),
    CryptoSignatureRule(r"crypto/sha1\.(New|Sum)", "SHA-1", "SHA-1", "hash", "SHA-1", ("SHA-1", "SHA1")),
    CryptoSignatureRule(r"crypto/md5\.(New|Sum)", "MD5", "MD5", "hash", aliases=("MD5",)),
    CryptoSignatureRule(r"crypto/hmac\.New", "HMAC", "HMAC", "message_authentication", aliases=("HMAC",)),
    CryptoSignatureRule(r"crypto/ed25519\.(Sign|Verify|GenerateKey)", "Ed25519", "Ed25519", "signature", aliases=("Ed25519",)),
    CryptoSignatureRule(r"crypto/tls\..*", "TLS", "TLS", "secure_transport", aliases=("TLS",)),
    CryptoSignatureRule(r"crypto/x509\..*", "X.509", "X.509", "certificate_processing", aliases=("X.509", "X509")),
    CryptoSignatureRule(r"golang\.org/x/crypto/chacha20poly1305\..*", "ChaCha20-Poly1305", "ChaCha20", "aead", aliases=("ChaCha20-Poly1305",)),
    CryptoSignatureRule(r"golang\.org/x/crypto/argon2\..*", "Argon2", "Argon2", "password_hash", aliases=("Argon2",)),
    CryptoSignatureRule(r"golang\.org/x/crypto/pbkdf2\..*", "PBKDF2", "PBKDF2", "key_derivation", aliases=("PBKDF2",)),
    CryptoSignatureRule(r"golang\.org/x/crypto/hkdf\..*", "HKDF", "HKDF", "key_derivation", aliases=("HKDF",)),
    CryptoSignatureRule(r"golang\.org/x/crypto/sha3\..*", "SHA-3", "SHA-3", "hash", aliases=("SHA-3", "SHA3")),
    CryptoSignatureRule(r"golang\.org/x/crypto/blake2(b|s)\..*", "BLAKE2", "BLAKE2", "hash", aliases=("BLAKE2b", "BLAKE2s")),
    CryptoSignatureRule(r"golang\.org/x/crypto/scrypt\..*", "scrypt", "scrypt", "password_hash", aliases=("scrypt",)),
    CryptoSignatureRule(r"golang\.org/x/crypto/bcrypt\..*", "bcrypt", "bcrypt", "password_hash", aliases=("bcrypt",)),
    CryptoSignatureRule(r"golang\.org/x/crypto/twofish\..*", "Twofish", "Twofish", "block_cipher", aliases=("Twofish",)),
    CryptoSignatureRule(r"github\.com/cloudflare/circl/kem/mlkem/.*", "ML-KEM", "ML-KEM", "key_encapsulation", aliases=("ML-KEM", "Kyber")),
    CryptoSignatureRule(r"github\.com/cloudflare/circl/sign/mldsa/.*", "ML-DSA", "ML-DSA", "signature", aliases=("ML-DSA", "Dilithium")),
    CryptoSignatureRule(r"github\.com/cloudflare/circl/sign/slhdsa/.*", "SLH-DSA", "SLH-DSA", "signature", aliases=("SLH-DSA", "SPHINCS+")),
    CryptoSignatureRule(r"crypto/mlkem\..*", "ML-KEM", "ML-KEM", "key_encapsulation", aliases=("ML-KEM",)),
)


def _crypto_signature_rule(symbol: str) -> CryptoSignatureRule | None:
    for rule in CRYPTO_SIGNATURE_RULES:
        if re.search(rule.symbol_pattern, symbol):
            return rule
    return None


@lru_cache(maxsize=1024)
def _term_pattern(term: str) -> re.Pattern[str]:
    parts = re.findall(r"[A-Za-z0-9]+", term)
    if not parts:
        return re.compile(r"(?!)")
    body = r"[\W_]*".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


@lru_cache(maxsize=128)
def _package_symbol_pattern(package_path: str) -> re.Pattern[str]:
    package = re.escape(package_path)
    return re.compile(
        rf"(?<![A-Za-z0-9_/]){package}(?:\.\(\*?[A-Za-z0-9_]+\))?\.[A-Za-z_][A-Za-z0-9_]*"
    )


def _text_hits(text: str, catalog: Iterable[tuple[str, str, tuple[str, ...]]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for canonical, category, aliases in catalog:
        matched = next((alias for alias in aliases if _term_pattern(alias).search(text)), None)
        if matched is not None:
            hits.append({"name": canonical, "category": category, "family": _algorithm_family(canonical),
                         "variant": canonical if _algorithm_family(canonical) != canonical else None,
                         "matched_text": matched})
    return hits


def _algorithm_family(name: str) -> str:
    if name.startswith("ML-KEM-") or name == "Kyber": return "ML-KEM"
    if name.startswith("ML-DSA-") or name == "FN-DSA": return "ML-DSA"
    if name.startswith("AES-"): return "AES"
    if name.startswith("HMAC-"): return "HMAC"
    if name.startswith("SHA3-") or name.startswith("SHAKE"): return "SHA-3"
    if name.startswith("SHA-"): return "SHA-2" if name != "SHA-1" else "SHA-1"
    if name == "RSA-PSS" or name == "RSA-OAEP": return "RSA"
    if name == "ChaCha20-Poly1305" or name == "XChaCha20": return "ChaCha20"
    return name


def analyze_crypto(strings: Iterable[Any], imports: dict[int, str],
                   instructions: Iterable[Any] = (), artifact_path: str | None = None,
                   file_version: str | None = None,
                   go_functions: dict[int, str] | None = None,
                   go_symbol_metadata: dict[str, Any] | None = None,
                   go_build_metadata: dict[str, Any] | None = None,
                   executable_ranges: Iterable[tuple[int, int]] | None = None,
                   application_reachable_functions: set[int] | None = None) -> dict[str, Any]:
    """Return explicitly classified static crypto indicators and binary identity."""
    instructions = list(instructions)
    findings: list[dict[str, Any]] = []
    algorithms: Counter[str] = Counter()
    keywords: Counter[str] = Counter()
    package_paths: Counter[str] = Counter()
    package_algorithms: set[str] = set()
    symbol_name_strings: dict[tuple[str, str], dict[str, Any]] = {}
    symbol_name_string_occurrences = 0
    go_function_symbols: list[dict[str, Any]] = []
    crypto_operations: list[dict[str, Any]] = []
    api_import_count = 0
    go_runtime_version: str | None = None
    crypto_module_dependencies = [
        dependency for dependency in (go_build_metadata or {}).get("dependencies", [])
        if any(marker in str(dependency.get("path", "")).lower() for marker in _CRYPTO_MODULE_MARKERS)
    ]

    for item in strings:
        text = str(getattr(item, "text", ""))
        if go_runtime_version is None:
            version_match = re.search(r"(?<![A-Za-z0-9])go1\.\d+(?:\.\d+)?(?![A-Za-z0-9.])", text)
            if version_match:
                go_runtime_version = version_match.group(0)
        try:
            raw_address = getattr(item, "address")
            address = hex(int(raw_address, 0) if isinstance(raw_address, str) else int(raw_address))
        except (TypeError, ValueError):
            address = None
        # Ignore embedded scanner/filter regexes to prevent false-positive self-referential hits
        is_scanner_regex = (
            text.startswith("(?i)")
            or r"\b(" in text
            or (r"[-_]?" in text and ("|" in text or "(" in text))
        )
        algorithm_hits = [] if is_scanner_regex else _text_hits(text, _ALGORITHMS)
        keyword_hits = [] if is_scanner_regex else _text_hits(text, _KEYWORDS)
        package_hits = [
            {"package_path": package_path, "algorithm": algorithm}
            for package_path, algorithm in _CRYPTO_PACKAGE_PATHS
            if _term_pattern(package_path).search(text)
        ]
        for hit in package_hits:
            package_paths[hit["package_path"]] += 1
            if hit["algorithm"] != "cipher API":
                package_algorithms.add(hit["algorithm"])
        symbol_hits: list[dict[str, str]] = []
        for package_path, algorithm in _CRYPTO_PACKAGE_PATHS:
            for match in _package_symbol_pattern(package_path).finditer(text):
                symbol = match.group(0)
                symbol_hits.append({"package_path": package_path, "algorithm": algorithm, "symbol": symbol})
                key = (package_path, symbol)
                entry = symbol_name_strings.setdefault(key, {
                    "package_path": package_path, "algorithm": algorithm,
                    "symbol": symbol, "occurrence_count": 0, "addresses": [],
                })
                entry["occurrence_count"] += 1
                if address is not None and address not in entry["addresses"] and len(entry["addresses"]) < 16:
                    entry["addresses"].append(address)
                symbol_name_string_occurrences += 1
        for hit in algorithm_hits:
            algorithms[hit["name"]] += 1
        for hit in keyword_hits:
            keywords[hit["name"]] += 1
        display_text = text[:256] if len(text) > 256 else text
        if algorithm_hits or keyword_hits:
            findings.append({
                "evidence_type": "string_match",
                "confidence": "low",
                "address": address,
                "section": getattr(item, "section", None),
                "encoding": getattr(item, "encoding", None),
                "text": display_text,
                "algorithms": algorithm_hits,
                "keywords": keyword_hits,
                "interpretation": "A cryptography-related name or keyword is present in a readable binary string.",
            })
        for hit in package_hits:
            findings.append({
                "evidence_type": "go_package_path",
                "confidence": "low",
                "address": address,
                "section": getattr(item, "section", None),
                "text": display_text,
                "algorithm": hit["algorithm"],
                "package_path": hit["package_path"],
                "interpretation": "A Go package path string is present; this alone does not prove code is linked or executed.",
            })
        for hit in symbol_hits:
            findings.append({
                "evidence_type": "go_symbol_name_string",
                "confidence": "low",
                "address": address,
                "section": getattr(item, "section", None),
                "algorithm": hit["algorithm"],
                "package_path": hit["package_path"],
                "symbol": hit["symbol"],
                "interpretation": "Text shaped like a package-qualified Go function symbol is present; this is string evidence until confirmed in the Go pclntab function table.",
            })

    go_symbol_names = dict(go_functions or {})
    for function_address, symbol in sorted(go_symbol_names.items()):
        for package_path, algorithm in _CRYPTO_PACKAGE_PATHS:
            if symbol.startswith(package_path + "."):
                rule = _crypto_signature_rule(symbol)
                if rule is not None and rule.algorithm is not None:
                    algorithm = rule.algorithm
                row = {"address": hex(function_address), "symbol": symbol,
                       "package_path": package_path, "algorithm": algorithm,
                       "operation": rule.operation if rule else "cryptographic_primitive",
                       "algorithm_family": (rule.family if rule else _algorithm_family(algorithm)) if algorithm != "cipher API" else None,
                       "variant": rule.variant if rule else None,
                       "aliases": list(rule.aliases) if rule else [],
                       "evidence_requirements": list(rule.evidence_requirements) if rule else ["go_function_symbol"]}
                go_function_symbols.append(row)
                if rule is not None:
                    operation_row = {key: row[key] for key in (
                        "address", "symbol", "package_path", "algorithm", "operation",
                        "algorithm_family", "variant", "aliases", "evidence_requirements")}
                    crypto_operations.append(operation_row)
                    findings.append({"evidence_type": "go_crypto_operation_symbol", **operation_row,
                                     "confidence": "medium",
                                     "interpretation": "A pclntab function symbol matches a named crypto operation rule; it does not prove call or execution."})
                findings.append({"evidence_type": "go_function_symbol", **row,
                                 "confidence": "medium",
                                 "interpretation": "Go pclntab maps this symbol to a function entry in executable code."})
                break

    for address, symbol in sorted(imports.items()):
        lowered = symbol.lower()
        if not any(marker.lower() in lowered for marker in _CRYPTO_IMPORT_MARKERS):
            continue
        api_import_count += 1
        findings.append({
            "evidence_type": "imported_api",
            "confidence": "medium",
            "address": hex(address),
            "symbol": symbol,
            "interpretation": "The executable imports a cryptography-related API; this does not establish that it is called.",
        })

    imported_crypto_addresses = {
        address for address, symbol in imports.items()
        if any(marker.lower() in symbol.lower() for marker in _CRYPTO_IMPORT_MARKERS)
    }
    callsite_count = 0
    symbol_starts = sorted(go_symbol_names)
    symbol_ends = {start: (symbol_starts[i + 1] if i + 1 < len(symbol_starts) else None)
                   for i, start in enumerate(symbol_starts)}
    code_ranges = tuple(executable_ranges or ())

    def containing_go_function(address: int) -> tuple[int, str] | None:
        # pclntab function entries are ordered by PC. The next entry bounds the
        # current function for attribution purposes; gaps are conservatively
        # accepted because Go text may contain alignment padding between funcs.
        import bisect
        if code_ranges and not any(start <= address < end for start, end in code_ranges):
            return None
        index = bisect.bisect_right(symbol_starts, address) - 1
        if index < 0:
            return None
        start = symbol_starts[index]
        end = symbol_ends[start]
        if end is None and code_ranges:
            end = next((bound for low, bound in code_ranges if low <= start < bound), None)
        return (start, go_symbol_names[start]) if end is None or address < end else None

    call_graph_edges: list[dict[str, Any]] = []
    unresolved_indirect_calls: list[dict[str, Any]] = []
    adjacency: dict[int, set[int]] = {}
    for ins in instructions:
        if getattr(ins, "flow", None) != "call":
            continue
        caller = containing_go_function(int(ins.address))
        direct_target = getattr(ins, "direct_target", None)
        raw_targets = ([direct_target] if direct_target is not None
                       else list(getattr(ins, "indirect_call_targets", None) or ()))
        targets = sorted({int(target) for target in raw_targets if target is not None})
        if not targets and direct_target is None and not getattr(ins, "import_name", None):
            unresolved_indirect_calls.append({
                "evidence_type": "unresolved_indirect_call", "address": hex(int(ins.address)),
                "state": "unknown", "confidence": "low",
                "caller_address": hex(caller[0]) if caller else None,
                "caller": caller[1] if caller else None,
                "instruction": getattr(ins, "text", None), "operand": getattr(ins, "operands", None),
                "reason": getattr(ins, "unresolved_call_reason", None) or "no concrete function target was established",
                "resolution": "unresolved",
                "interpretation": "Indirect call target was not resolved by current disassembly or data-flow evidence; no caller-to-callee edge is asserted.",
            })
        unresolved_targets = []
        for target in targets:
            callee = containing_go_function(target)
            if callee is None:
                unresolved_targets.append(hex(target))
                continue
            if caller is None:
                continue
            kind = "direct" if direct_target is not None else "resolved_indirect"
            provenance = "direct_branch_target" if direct_target is not None else getattr(ins, "resolved_call_provenance", None) or "static_target_analysis"
            adjacency.setdefault(caller[0], set()).add(callee[0])
            call_graph_edges.append({
                "call_address": hex(int(ins.address)), "caller_address": hex(caller[0]),
                "caller": caller[1], "target_address": hex(target),
                "callee_address": hex(callee[0]), "callee": callee[1], "call_kind": kind,
                "resolution_provenance": provenance,
            })
        if direct_target is None and targets and (unresolved_targets or caller is None):
            unresolved_indirect_calls.append({
                "evidence_type": "unresolved_indirect_call", "address": hex(int(ins.address)),
                "state": "unknown", "confidence": "low",
                "caller_address": hex(caller[0]) if caller else None,
                "caller": caller[1] if caller else None,
                "instruction": getattr(ins, "text", None),
                "target_candidates": [hex(target) for target in targets],
                "unmapped_target_candidates": unresolved_targets,
                "resolution": "partially_resolved",
                "interpretation": "Concrete target candidates exist, but one or more cannot be mapped to a recovered Go function; no edge is asserted for unmapped candidates.",
            })

    roots = {address for address, symbol in go_symbol_names.items()
             if symbol in {"runtime.main", "main.main"} or
             (symbol.endswith(".init") and not symbol.startswith("runtime."))}
    reachable: set[int] = set(roots)
    pending = list(roots)
    while pending:
        current = pending.pop()
        for target in adjacency.get(current, ()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)

    reachable_edges = {(int(edge["caller_address"], 16), int(edge["callee_address"], 16))
                       for edge in call_graph_edges
                       if int(edge["caller_address"], 16) in reachable}
    for ins in instructions:
        if getattr(ins, "flow", None) != "call":
            continue
        target = getattr(ins, "direct_target", None)
        import_name = getattr(ins, "import_name", None)
        imported_target = target in imported_crypto_addresses or bool(
            import_name and any(marker.lower() in import_name.lower() for marker in _CRYPTO_IMPORT_MARKERS)
        )
        targets = set(getattr(ins, "indirect_call_targets", None) or ())
        if target is not None:
            targets.add(target)
        for resolved_target in sorted(targets):
            resolved_go_function = containing_go_function(resolved_target)
            go_target_name = resolved_go_function[1] if resolved_go_function else None
            go_symbol_row = next((row for row in go_function_symbols
                                  if resolved_go_function and int(row["address"], 16) == resolved_go_function[0]), None)
            go_algorithm = go_symbol_row["algorithm"] if go_symbol_row else None
            if not imported_target and (go_algorithm is None or go_algorithm == "cipher API"):
                continue
            callsite_count += 1
            caller_info = containing_go_function(int(getattr(ins, "address")))
            direct_to_go_symbol = (target == resolved_target and resolved_go_function is not None)
            resolved_indirect = target is None and resolved_go_function is not None
            caller_entry = caller_info[0] if caller_info else None
            callee_entry = resolved_go_function[0] if resolved_go_function else resolved_target
            findings.append({
                "evidence_type": "disassembly_call",
                "confidence": "stronger_static_evidence" if direct_to_go_symbol or resolved_indirect else "medium",
                "address": hex(int(getattr(ins, "address"))),
                "target_address": hex(int(resolved_target)),
                "target_symbol": go_target_name or import_name or imports.get(resolved_target),
                "algorithm": go_algorithm,
                "operation": go_symbol_row.get("operation") if go_symbol_row else None,
                "variant": go_symbol_row.get("variant") if go_symbol_row else None,
                "call_kind": "direct" if direct_to_go_symbol else "resolved_indirect" if resolved_indirect else "resolved_static_target",
                "resolution_provenance": "direct_branch_target" if direct_to_go_symbol else getattr(ins, "resolved_call_provenance", None) or "static_target_analysis",
                "caller_address": hex(caller_entry) if caller_entry is not None else None,
                "caller": caller_info[1] if caller_info else None,
                "callee": go_target_name,
                "caller_reachable_from_application_roots": (
                    True if caller_entry in (application_reachable_functions or set())
                    else caller_entry in reachable if caller_entry is not None and roots else None
                ),
                "reachable_edge": (
                    caller_entry in (application_reachable_functions or set())
                    and (callee_entry in (application_reachable_functions or set()) or callee_entry in reachable)
                    if caller_entry is not None else False
                ) or ((caller_entry, callee_entry) in reachable_edges if caller_entry is not None else False),
                "instruction": getattr(ins, "text", None),
                "interpretation": "A decoded call instruction targets a resolved Go function symbol or identified imported crypto API. Indirect targets require concrete disassembly/data-flow evidence. Reachability is static and does not prove runtime execution.",
            })
        if imported_target and not targets:
            callsite_count += 1
            findings.append({"evidence_type": "disassembly_call",
                             "confidence": "medium",
                             "address": hex(int(getattr(ins, "address"))),
                             "target_address": hex(int(target)) if target is not None else None,
                             "target_symbol": import_name or imports.get(target), "algorithm": None,
                             "instruction": getattr(ins, "text", None),
                             "interpretation": "A decoded call instruction references an identified imported crypto API; this is static call-site evidence, not proof the path executes."})

    if go_runtime_version is None and go_build_metadata:
        candidate_version = go_build_metadata.get("toolchain")
        if isinstance(candidate_version, str) and candidate_version.startswith("go"):
            go_runtime_version = candidate_version
    artifact_identity = None
    if artifact_path:
        binary_path = Path(artifact_path)
        digest = hashlib.sha256()
        with binary_path.open("rb") as binary:
            for chunk in iter(lambda: binary.read(1024 * 1024), b""):
                digest.update(chunk)
        artifact_identity = {
            "filename": binary_path.name,
            "sha256": digest.hexdigest(),
            "file_size_bytes": binary_path.stat().st_size,
            "file_version": file_version,
            "go_runtime_version": go_runtime_version,
        }

    findings.extend(unresolved_indirect_calls)
    symbol_algorithm_names = sorted({row["algorithm"] for row in go_function_symbols if row["algorithm"] != "cipher API"})
    string_algorithm_names = sorted(algorithms)
    grouped_evidence: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        if finding["evidence_type"] == "string_match":
            for hit in finding.get("algorithms", []):
                grouped_evidence.setdefault(hit["name"], []).append({
                    "type": "string_match", "confidence": "low", "address": finding.get("address"),
                    "section": finding.get("section"), "text": finding.get("text"),
                })
        elif finding["evidence_type"] == "go_function_symbol":
            grouped_evidence.setdefault(finding["algorithm"], []).append({
                "type": "go_function_symbol", "confidence": "medium", "address": finding["address"],
                "symbol": finding["symbol"], "package_path": finding["package_path"],
                "operation": finding.get("operation"), "variant": finding.get("variant"),
                "aliases": finding.get("aliases", []),
            })
        elif finding["evidence_type"] == "go_package_path" and finding.get("algorithm") != "cipher API":
            grouped_evidence.setdefault(finding["algorithm"], []).append({
                "type": "go_package_path", "confidence": "low", "address": finding.get("address"),
                "package_path": finding.get("package_path"),
            })
        elif finding["evidence_type"] == "disassembly_call" and finding.get("algorithm"):
            grouped_evidence.setdefault(finding["algorithm"], []).append({
                "type": ("direct_call_site" if finding.get("call_kind") == "direct" else
                         "indirect_call_site" if finding.get("call_kind") == "resolved_indirect" else
                         "resolved_static_call_site"),
                "confidence": finding.get("confidence", "medium"), "address": finding.get("address"),
                "caller": finding.get("caller"), "callee": finding.get("callee"),
                "operation": finding.get("operation"), "variant": finding.get("variant"),
                "target_address": finding.get("target_address"),
                "caller_reachable_from_application_roots": finding.get("caller_reachable_from_application_roots"),
                "reachable_edge": finding.get("reachable_edge"),
            })
    state_names = set(grouped_evidence) | package_algorithms
    algorithm_findings = []
    for name in sorted(state_names):
        evidence = grouped_evidence.get(name, [])
        states = []
        has_string_evidence = (any(item["type"] in {"string_match", "go_package_path"} for item in evidence)
                               or name in package_algorithms)
        has_symbol_evidence = any(item["type"] == "go_function_symbol" for item in evidence)
        if has_string_evidence and not has_symbol_evidence:
            states.append("string_only")
        if has_symbol_evidence:
            states.append("symbol_present")
        call_evidence = [item for item in evidence if item["type"] in {
            "direct_call_site", "indirect_call_site", "resolved_static_call_site"
        }]
        if call_evidence:
            states.append("call_site_found")
        if any(item.get("reachable_edge") for item in call_evidence):
            states.append("statically_reachable")
        status = next((state for state in reversed(states)), "unknown")
        algorithm_findings.append({"algorithm": name, "family": _algorithm_family(name),
                                   "status": status, "states": states or ["unknown"],
                                   "runtime_observed": False, "runtime_state": "unknown",
                                   "evidence": evidence})
    call_graph_report = {
        "edge_count": len(call_graph_edges),
        "direct_edge_count": sum(edge["call_kind"] == "direct" for edge in call_graph_edges),
        "resolved_indirect_edge_count": sum(edge["call_kind"] == "resolved_indirect" for edge in call_graph_edges),
        "root_functions": [{"address": hex(address), "symbol": go_symbol_names[address]} for address in sorted(roots)],
        "reachable_function_count": len(reachable),
        "crypto_functions_reachable_from_roots": sorted(
            {go_symbol_names[target] for caller, target in reachable_edges
             if any(row["address"] == hex(target) and row["algorithm"] != "cipher API"
                    for row in go_function_symbols)}
        ),
        "edges": call_graph_edges,
        "unresolved_indirect_calls": unresolved_indirect_calls,
        "unresolved_indirect_call_count": len(unresolved_indirect_calls),
        "limitations": ["Indirect edges are reported only when existing disassembly/data-flow analysis yields concrete candidate targets.",
                        "Go interface dispatch and function values are not inferred from type metadata; unresolved calls remain explicit.",
                        "Inlining, callbacks whose targets are not call targets, and unresolved thunks are not included.",
                        "Reachability from recovered Go main/init roots is static and is not runtime observation."],
    }
    direct_graph_summary = {
        key: call_graph_report[key] for key in (
            "edge_count", "direct_edge_count", "resolved_indirect_edge_count",
            "root_functions", "reachable_function_count", "crypto_functions_reachable_from_roots",
            "unresolved_indirect_call_count", "limitations",
        )
    }
    return {
        "analyzer": {"name": ANALYZER_NAME, "version": ANALYZER_VERSION},
        "analysis_scope": "PE readable strings, imported API names, Go build metadata, pclntab function symbols, and statically resolved direct/indirect calls",
        "artifact_identity": artifact_identity,
        "runtime_observation": {
            "available": False,
            "reason": "This scanner performs static analysis and does not execute the target under instrumentation.",
        },
        "algorithm_names_found": sorted(algorithms),
        "algorithm_findings": algorithm_findings,
        "algorithm_use_states": {item["algorithm"]: item["states"] for item in algorithm_findings},
        "crypto_signature_rule_count": len(CRYPTO_SIGNATURE_RULES),
        "crypto_operations_found": crypto_operations,
        "algorithm_families": {
            family: sorted(name for name in algorithms if _algorithm_family(name) == family)
            for family in sorted({_algorithm_family(name) for name in algorithms})
        },
        "algorithm_evidence_tiers": {
            "string_name_candidates": string_algorithm_names,
            "go_function_symbol_names": symbol_algorithm_names,
            "string_only_candidates": sorted(set(string_algorithm_names) - set(symbol_algorithm_names)),
            "interpretation": "String-only candidates may be documentation, scanner rules, test data, or metadata; package-qualified Go function symbols are stronger evidence of code or symbol metadata in the binary, but neither proves runtime use.",
        },
        "algorithm_string_hit_counts": dict(sorted(algorithms.items())),
        "keyword_hit_counts": dict(sorted(keywords.items())),
        "algorithm_string_hits": sum(algorithms.values()),
        "keyword_string_hits": sum(keywords.values()),
        "cryptography_api_imports": api_import_count,
        "disassembly_call_sites_to_crypto_functions_or_imports": callsite_count,
        "go_call_graph": call_graph_report,
        "go_direct_call_graph": direct_graph_summary,
        "package_path_string_evidence": {
            "algorithm_names": sorted(package_algorithms),
            "package_path_counts": dict(sorted(package_paths.items())),
            "package_path_hit_count": sum(package_paths.values()),
            "interpretation": "Package path strings suggest linked or retained package metadata; they do not prove runtime execution.",
        },
        "go_pclntab": go_symbol_metadata or {"format": None, "function_count": len(go_symbol_names), "warnings": []},
        "go_build_metadata": go_build_metadata or {"available": False, "toolchain": go_runtime_version,
                                                     "module": None, "dependencies": [],
                                                     "build_settings": [], "warnings": ["Go build info was not supplied by the PE loader."]},
        "go_crypto_module_dependencies": {
            "modules": crypto_module_dependencies,
            "interpretation": "Go build metadata lists these crypto-related dependencies; module presence alone does not prove the package or algorithm is linked or used.",
        },
        "go_function_symbol_evidence": {
            "unique_symbols": len(go_function_symbols),
            "algorithm_names": symbol_algorithm_names,
            "symbols": go_function_symbols,
            "interpretation": "These names are parsed from the Go function table and map to executable entry addresses; presence does not prove a call path executes.",
        },
        "go_symbol_name_string_evidence": {
            "unique_symbols": len(symbol_name_strings),
            "symbol_occurrences": symbol_name_string_occurrences,
            "algorithm_names": sorted({row["algorithm"] for row in symbol_name_strings.values() if row["algorithm"] != "cipher API"}),
            "symbols": sorted(symbol_name_strings.values(), key=lambda row: (row["package_path"], row["symbol"])),
            "interpretation": "Package-like text in strings is not promoted to a function symbol unless it is confirmed in pclntab.",
        },
        "compiled_function_symbol_string_evidence": {
            "unique_symbols": len(symbol_name_strings),
            "symbol_occurrences": symbol_name_string_occurrences,
            "algorithm_names": sorted({row["algorithm"] for row in symbol_name_strings.values() if row["algorithm"] != "cipher API"}),
            "symbols": sorted(symbol_name_strings.values(), key=lambda row: (row["package_path"], row["symbol"])),
            "interpretation": "These are symbol-shaped strings and remain string evidence unless independently recovered from pclntab.",
        },
        "finding_count": len(findings),
        "findings": findings,
        "limitations": [
            "A string hit shows that text is present in the binary; it does not prove the algorithm is used.",
            "An imported crypto API indicates a dependency or capability, not the exact algorithm selected at runtime.",
            "Names hidden by packing, encryption, dynamic construction, or constants are not detected by this string catalog.",
            "The catalog includes common classical, post-quantum, hybrid, and MAC variants, but is not exhaustive; absence of a hit is not evidence that the binary has no cryptography.",
            "Short names such as DSA and LMS can be ambiguous in strings; review each matched string and its address before treating it as an algorithm implementation.",
        ],
    }


def build_crypto_cbom(input_path: str, report: dict[str, Any]) -> dict[str, Any]:
    """Build a CycloneDX 1.7 CBOM with evidence and algorithm variants preserved."""

    path = Path(input_path)
    artifact_identity = report.get("artifact_identity") or {}
    by_name: dict[str, dict[str, Any]] = {}
    category_by_name = {name: category for name, category, _ in _ALGORITHMS}
    primitive_by_category = {
        "public_key": "pke", "signature": "signature", "key_exchange": "key-agree",
        "post_quantum_kem": "kem", "post_quantum_signature": "signature",
        "hybrid_key_encapsulation": "kem", "hybrid_protocol": None,
        "symmetric_cipher": "block-cipher", "authenticated_cipher": "ae",
        "block_cipher_mode": "block-cipher", "stream_cipher": "stream-cipher",
        "hash": "hash", "hash_family": "hash", "extendable_output_hash": "xof",
        "message_authentication": "mac", "key_derivation": "kdf", "password_hash": "hash",
        "protocol": None, "certificate_format": None, "standard": "other",
    }
    cyclone_families = {
        "3DES", "AES", "BLAKE2", "BLAKE3", "Blowfish", "ChaCha", "ChaCha20",
        "CMAC", "DES", "DSA", "ECDH", "ECDSA", "EdDSA", "HKDF", "HMAC", "LMS",
        "MD5", "ML-DSA", "ML-KEM", "PBKDF2", "Poly1305", "RC4", "RSAES-OAEP",
        "RSASSA-PSS", "SHA-1", "SHA-2", "SHA-3", "SLH-DSA", "SM4", "Twofish", "XMSS",
        "bcrypt",
    }

    def component(name: str) -> dict[str, Any]:
        if name not in by_name:
            safe_ref = re.sub(r"[^a-z0-9.-]+", "-", name.lower()).strip("-") or "unknown"
            category = category_by_name.get(name, "standard")
            asset_type = "protocol" if category in {"protocol", "hybrid_protocol"} else (
                "certificate" if category == "certificate_format" else "algorithm"
            )
            primitive = primitive_by_category.get(category)
            crypto_properties: dict[str, Any] = {"assetType": asset_type}
            if asset_type == "algorithm" and primitive:
                algorithm_properties: dict[str, str] = {"primitive": primitive}
                family = _algorithm_family(name)
                if family in cyclone_families:
                    algorithm_properties["algorithmFamily"] = family
                crypto_properties["algorithmProperties"] = algorithm_properties
            by_name[name] = {
                "type": "cryptographic-asset",
                "bom-ref": f"crypto-{safe_ref}",
                "name": name,
                "cryptoProperties": crypto_properties,
                "evidence": {"occurrences": []},
                "properties": [{"name": "crypto:algorithmFamily", "value": _algorithm_family(name)}],
            }
        return by_name[name]

    for finding in report.get("findings", []):
        address = finding.get("address")
        section = finding.get("section")
        if finding.get("evidence_type") == "string_match":
            for hit in finding.get("algorithms", []):
                row = component(hit["name"])
                row["evidence"]["occurrences"].append({
                    "location": path.name,
                    "symbol": finding.get("text", ""),
                    "additionalContext": json.dumps({
                        "evidence_type": "string_match", "confidence": finding.get("confidence", "low"), "address": address, "section": section,
                        "interpretation": finding["interpretation"],
                    }, ensure_ascii=False),
                })
        if finding.get("evidence_type") == "go_function_symbol":
            row = component(finding["algorithm"])
            row["evidence"]["occurrences"].append({
                "location": path.name,
                "symbol": finding["symbol"],
                "additionalContext": json.dumps({
                    "evidence_type": "go_function_symbol", "confidence": finding.get("confidence", "medium"), "package_path": finding["package_path"],
                    "address": address, "operation": finding.get("operation"),
                    "variant": finding.get("variant"), "aliases": finding.get("aliases", []),
                    "interpretation": finding["interpretation"],
                }, ensure_ascii=False),
            })
        if finding.get("evidence_type") == "go_package_path" and finding.get("algorithm") != "cipher API":
            row = component(finding["algorithm"])
            row["evidence"]["occurrences"].append({
                "location": path.name,
                "symbol": finding["package_path"],
                "additionalContext": json.dumps({
                    "evidence_type": "go_package_path", "confidence": finding.get("confidence", "low"), "address": address,
                    "interpretation": finding["interpretation"],
                }, ensure_ascii=False),
            })
        if finding.get("evidence_type") == "disassembly_call" and finding.get("algorithm"):
            row = component(finding["algorithm"])
            row["evidence"]["occurrences"].append({
                "location": path.name,
                "symbol": finding.get("target_symbol", ""),
                "additionalContext": json.dumps({
                    "evidence_type": "disassembly_call", "call_address": address,
                    "confidence": finding.get("confidence", "medium"),
                    "operation": finding.get("operation"), "variant": finding.get("variant"),
                    "target_address": finding.get("target_address"),
                    "call_kind": finding.get("call_kind"),
                    "caller_address": finding.get("caller_address"),
                    "caller": finding.get("caller"),
                    "callee": finding.get("callee"),
                    "caller_reachable_from_application_roots": finding.get("caller_reachable_from_application_roots"),
                    "reachable_edge": finding.get("reachable_edge"),
                    "instruction": finding.get("instruction"),
                    "interpretation": finding["interpretation"],
                }, ensure_ascii=False),
            })

    components = []
    for name, row in sorted(by_name.items()):
        occurrences = row["evidence"]["occurrences"]
        contexts = [json.loads(occ["additionalContext"]) for occ in occurrences]
        string_count = sum(1 for context in contexts if context["evidence_type"] == "string_match")
        package_count = sum(1 for context in contexts if context["evidence_type"] == "go_package_path")
        symbol_count = sum(1 for context in contexts if context["evidence_type"] == "go_function_symbol")
        call_count = sum(1 for context in contexts if context["evidence_type"] == "disassembly_call")
        evidence_states = []
        if (string_count or package_count) and not symbol_count: evidence_states.append("string_only")
        if symbol_count: evidence_states.append("symbol_present")
        if call_count: evidence_states.append("call_site_found")
        if any(context.get("reachable_edge") for context in contexts): evidence_states.append("statically_reachable")
        row["properties"] = [
            {"name": "org.independent-decompiler.evidence.string_occurrences", "value": str(string_count)},
            {"name": "org.independent-decompiler.evidence.go_package_path_occurrences", "value": str(package_count)},
            {"name": "org.independent-decompiler.evidence.go_function_symbols", "value": str(symbol_count)},
            {"name": "org.independent-decompiler.evidence.disassembly_calls", "value": str(call_count)},
            {"name": "org.independent-decompiler.evidence.states", "value": ",".join(evidence_states) or "unknown"},
            {"name": "org.independent-decompiler.evidence.runtime_observed", "value": "false"},
        ]
        components.append(row)

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "tools": {"components": [{"type": "application", "name": ANALYZER_NAME, "version": ANALYZER_VERSION}]},
            "component": {
                "type": "file",
                "name": path.name,
                "hashes": ([{"alg": "SHA-256", "content": artifact_identity["sha256"]}]
                           if artifact_identity.get("sha256") else []),
                "properties": [{"name": "analysis.scope", "value": "static PE strings, imports, Go build metadata, pclntab function symbols, and resolved direct/indirect call sites"}],
            },
            "properties": [
                {"name": "analysis.warning", "value": "String matches and linked symbols do not prove runtime execution."},
                {"name": "analysis.crypto_api_import_count", "value": str(report.get("cryptography_api_imports", 0))},
                {"name": "analysis.artifact.file_size_bytes", "value": str(artifact_identity.get("file_size_bytes", "unknown"))},
                {"name": "analysis.artifact.file_version", "value": artifact_identity.get("file_version") or "not embedded"},
                {"name": "analysis.artifact.go_runtime_version", "value": artifact_identity.get("go_runtime_version") or "not found"},
                {"name": "analysis.analyzer.version", "value": ANALYZER_VERSION},
                {"name": "analysis.disassembly_crypto_call_site_count", "value": str(report.get("disassembly_call_sites_to_crypto_functions_or_imports", 0))},
                {"name": "analysis.go_build_metadata_available", "value": str(bool(report.get("go_build_metadata", {}).get("available", False))).lower()},
                {"name": "analysis.go_module_path", "value": str((report.get("go_build_metadata", {}).get("module") or {}).get("path", "unknown"))},
                {"name": "analysis.go_dependency_count", "value": str(len(report.get("go_build_metadata", {}).get("dependencies", [])))},
                {"name": "analysis.go_dependency_paths", "value": ", ".join(dep.get("path", "") for dep in report.get("go_build_metadata", {}).get("dependencies", []) if dep.get("path")) or "none"},
                {"name": "analysis.go_crypto_dependency_modules", "value": ", ".join(dep.get("path", "") for dep in report.get("go_crypto_module_dependencies", {}).get("modules", [])) or "none"},
                {"name": "analysis.go_direct_call_graph_edge_count", "value": str(report.get("go_direct_call_graph", {}).get("edge_count", 0))},
                {"name": "analysis.go_crypto_functions_reachable_from_roots", "value": ", ".join(report.get("go_direct_call_graph", {}).get("crypto_functions_reachable_from_roots", [])) or "none confirmed"},
                {"name": "analysis.go_resolved_indirect_edge_count", "value": str(report.get("go_direct_call_graph", {}).get("resolved_indirect_edge_count", 0))},
                {"name": "analysis.go_unresolved_indirect_call_count", "value": str(report.get("go_direct_call_graph", {}).get("unresolved_indirect_call_count", 0))},
                {"name": "analysis.runtime_observed", "value": "false"},
                {"name": "analysis.limitations", "value": "; ".join(
                    report.get("limitations", []) + report.get("go_call_graph", {}).get("limitations", [])
                )},
            ],
        },
        "components": components,
    }
