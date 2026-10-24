#!/usr/bin/env python3
"""Check the actual packaged files against this source checkout before publishing."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import tarfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHS = [75, 80, 86, 89, 90, 120]


def text(data):
    return data.decode("utf-8-sig").replace("\r\n", "\n")


def check_notices(files, version, release_notes=True):
    names = ["LICENSE", "NOTICE", "README.md", "USAGE.md", "vectors.json"]
    if release_notes:
        names += ["PLATFORMS.md", "RELEASE-v" + version + ".md"]
    for name in names:
        assert name in files, "missing packaged " + name
        assert text(files[name]) == text((ROOT / name).read_bytes()), "source mismatch: " + name
    license_text = text(files["LICENSE"])
    assert license_text.startswith("ProxiChorizo License 1.0\n")
    assert "written permission" in license_text and "Commercial" in license_text


def read_archive(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            files = {i.filename.replace("\\", "/"): z.read(i) for i in z.infolist() if not i.is_dir()}
    else:
        with tarfile.open(path) as t:
            members = t.getmembers()
            assert all(m.isfile() or m.isdir() for m in members), "unexpected link or special file"
            files = {m.name: t.extractfile(m).read() for m in members if m.isfile()}
    for name in files:
        p = pathlib.PurePosixPath(name)
        assert not p.is_absolute() and ".." not in p.parts, "unsafe archive member"
    roots = [name[:-len("LICENSE")] for name in files if name.endswith("/LICENSE") or name == "LICENSE"]
    assert len(roots) == 1, "expected one package root"
    prefix = roots[0]
    assert all(name.startswith(prefix) for name in files), "files outside package root"
    return {name[len(prefix):]: data for name, data in files.items()}


def verify_archive(path, version, commit):
    files = read_archive(path)
    check_notices(files, version)
    info = json.loads(text(files["BUILD-INFO.json"]))
    assert info["version"] == version and info["source_commit"] == commit
    assert info["architectures"] == ARCHS, "architecture metadata mismatch"
    expected = {}
    for line in text(files["SHA256SUMS.txt"]).splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        assert match, "invalid internal checksum line"
        name = match[2].replace("\\", "/")
        if name.startswith("./"):
            name = name[2:]
        assert name not in expected, "duplicate internal checksum"
        expected[name] = match[1].lower()
    assert set(expected) == set(files) - {"SHA256SUMS.txt"}, "incomplete checksum coverage"
    for name, digest in expected.items():
        assert hashlib.sha256(files[name]).hexdigest() == digest, "checksum mismatch: " + name
    if path.suffix == ".zip":
        assert info["platform"] == "native-windows-x86_64"
        assert info["windows_gpu_hardware_test"] is False
        assert files["proxichorizo.exe"].startswith(b"MZ")
        assert "ProxiChorizo " + version in text(files["README.txt"])
        for name in ["start.cmd", "start.ps1", "README.txt"]:
            assert name in files
    else:
        assert info["platform"] == "linux-x86_64" and info["opencl"] is True
        for name in ["proxichorizo", "proxichorizo-cl"]:
            assert files[name].startswith(b"\x7fELF")
        for name in ["h-config.sh", "h-run.sh", "h-stats.sh", "mmpos-run.sh", "mmp-launch.sh", "mmp-stats.sh", "smos-run.sh"]:
            assert name in files
        assert "CUSTOM_VERSION=" + version in text(files["h-manifest.conf"])
        assert 'EXTERNAL_VERSION="' + version + '"' in text(files["mmp-external.conf"])
    sidecar = path.with_name(path.name + ".sha256")
    digest, name = text(sidecar.read_bytes()).strip().split(None, 1)
    assert name.lstrip(" *") == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest.lower()
    print("verified", path.name, "notices, version, architecture metadata and every checksum")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--directory", type=pathlib.Path)
    parser.add_argument("archives", type=pathlib.Path, nargs="*")
    args = parser.parse_args()
    assert args.directory or args.archives, "nothing to verify"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.directory:
        files = {p.name: p.read_bytes() for p in args.directory.iterdir() if p.is_file()}
        check_notices(files, args.version, release_notes=False)
        assert files["proxichorizo"].startswith(b"\x7fELF")
        print("verified container contents and notices against source", commit)
    for archive in args.archives:
        verify_archive(archive, args.version, commit)


if __name__ == "__main__":
    main()
