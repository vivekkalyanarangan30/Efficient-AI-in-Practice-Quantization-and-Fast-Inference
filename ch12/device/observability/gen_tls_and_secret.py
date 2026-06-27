"""Generate self-signed P-256 TLS cert + a CSPRNG basic-auth password.

Idempotent: refuses to overwrite existing material. Cert is for SAN
`localhost` + `127.0.0.1` only — bound to loopback, never published.

Outputs (all inside ch12/observability):
  tls/server.crt        # PEM, 0644
  tls/server.key        # PEM, 0600
  secrets/push.password # plaintext password, 0600  (consumed once into env)
  secrets/push.bcrypt   # bcrypt(cost=12) hash, 0600 (for pushgateway web.yml)
  secrets/push.user     # username, 0600
"""
from __future__ import annotations

import os
import secrets
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

BASE = Path(__file__).resolve().parent
TLS_DIR = BASE / "tls"
SEC_DIR = BASE / "secrets"


def _write_restricted(path: Path, data: bytes, mode: int = 0o600) -> None:
    if path.exists():
        # Honor §1: never silently overwrite secret material — fail closed.
        print(f"{path} already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(2)
    # Open exclusively with restrictive mode, then write.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    finally:
        os.chmod(path, mode)


def _make_cert() -> tuple[bytes, bytes]:
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(__import__("ipaddress").IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True, key_agreement=False,
                content_commitment=False, data_encipherment=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    crt_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return crt_pem, key_pem


def main() -> None:
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(TLS_DIR, 0o700)
    os.chmod(SEC_DIR, 0o700)

    crt_pem, key_pem = _make_cert()
    _write_restricted(TLS_DIR / "server.crt", crt_pem, 0o644)
    _write_restricted(TLS_DIR / "server.key", key_pem, 0o600)

    # CSPRNG-generated 32-byte URL-safe password (~256 bits of entropy).
    password = secrets.token_urlsafe(32)
    username = "ch12_device"
    bhash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))

    _write_restricted(SEC_DIR / "push.user",     username.encode("utf-8") + b"\n", 0o600)
    _write_restricted(SEC_DIR / "push.password", password.encode("utf-8") + b"\n", 0o600)
    _write_restricted(SEC_DIR / "push.bcrypt",   bhash + b"\n", 0o600)

    print(f"wrote {TLS_DIR/'server.crt'} ({len(crt_pem)} bytes)")
    print(f"wrote {TLS_DIR/'server.key'} (perm=600)")
    print(f"wrote secrets/push.{{user,password,bcrypt}} (perm=600)")
    print("username:", username)
    print("(password kept on disk only because pushgateway can't read it from env)")


if __name__ == "__main__":
    main()
