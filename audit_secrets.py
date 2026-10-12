# audit_secrets.py

import argparse
import hashlib
import os
import re
import json
import sys
from collections import namedtuple
from typing import Iterable, List, Optional, Set
import math

# --- CONFIGURATION ---

ALLOWED_EXTENSIONS = {'.py', '.json', '.yaml', '.yml', '.env', '.txt', '.md'}
SCAN_FOLDERS = ['config', 'core', 'modules', 'skills']
SCAN_FILES = ['api.py', 'main.py', 'web_ui.py']
IGNORE_FOLDERS = {'.venv', '__pycache__', 'models'}
DATA_FOLDER = 'data'
MAX_DATA_FILE_SIZE = 5 * 1024 * 1024  # 5MB

ALLOWLIST = {
    'password': ['password123', 'changeme', 'example'],
    'token': ['your_token_here'],
    'api_key': ['your_api_key'],
    # Add more safe constants as needed
}

# ``fingerprint`` identifies a finding by file + type + the full line text (not
# the line number), so a baseline entry survives unrelated edits above it.
Secret = namedtuple('Secret', ['file', 'line', 'type', 'confidence', 'preview', 'fingerprint'],
                    defaults=[''])

# Backlog #197: a line carrying this marker is skipped (use sparingly, and
# only for values that are public by design).
IGNORE_MARKER = 'audit: ignore'
DEFAULT_BASELINE = '.secrets_baseline.json'
CONFIDENCE_RANK = {'low': 1, 'medium': 2, 'high': 3}

# --- REGEX PATTERNS ---

PATTERNS = [
    # API keys
    (re.compile(r'(?i)(openai|huggingface)[\w_]*[=:]\s*[\'"]?([a-z0-9\-]{20,})[\'"]?'), 'API Key', 'high'),
    (re.compile(r'(?i)(api[_-]?key|access[_-]?key|secret)[\w_]*[=:]\s*[\'"]?([a-z0-9\-]{20,})[\'"]?'), 'API Key', 'medium'),
    # AWS keys
    (re.compile(r'AKIA[0-9A-Z]{16}'), 'AWS Access Key', 'high'),
    # Private keys
    (re.compile(r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'), 'Private Key', 'high'),
    # JWT tokens
    (re.compile(r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9._-]{10,}\.[a-zA-Z0-9._-]{10,}'), 'JWT Token', 'high'),
    # Hardcoded passwords/secrets/tokens
    (re.compile(r'(?i)(password|secret|token)[\w_]*[=:]\s*[\'"]?([^\'"\s]{6,})[\'"]?'), 'Hardcoded Secret', 'medium'),
    # URLs with embedded credentials
    (re.compile(r'https?://[^/\s:@]+:[^/\s:@]+@[^/\s]+'), 'URL with Credentials', 'high'),
]

# --- UTILITY FUNCTIONS ---



def fingerprint_for(filepath: str, secret_type: str, line: str) -> str:
    norm = filepath.replace('\\', '/')
    raw = f"{norm}|{secret_type}|{line.strip()}"
    return hashlib.sha1(raw.encode('utf-8', 'ignore')).hexdigest()[:16]


def is_allowed_file(filename: str) -> bool:
    return any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)

def is_allowlisted(key: str, value: str) -> bool:
    for k, vals in ALLOWLIST.items():
        if k in key.lower() and value in vals:
            return True
    return False

def shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    entropy = 0.0
    for x in set(data):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy -= p_x * math.log2(p_x)
    return entropy

def scan_file(filepath: str) -> List[Secret]:
    secrets = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                if IGNORE_MARKER in line:
                    continue
                for pattern, secret_type, confidence in PATTERNS:
                    for match in pattern.finditer(line):
                        key = match.group(1) if match.lastindex and match.lastindex >= 1 else ''
                        value = match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(0)
                        if is_allowlisted(key, value):
                            continue
                        preview = line.strip()[:50]
                        secrets.append(Secret(filepath, i, secret_type, confidence, preview,
                                              fingerprint_for(filepath, secret_type, line)))
                # Entropy-based detection for unknown secrets
                for word in re.findall(r'([A-Za-z0-9+/=]{20,})', line):
                    if len(word) > 20 and shannon_entropy(word) > 4.0 and not is_allowlisted('', word):
                        preview = line.strip()[:50]
                        secrets.append(Secret(filepath, i, 'High-Entropy String', 'low', preview,
                                              fingerprint_for(filepath, 'High-Entropy String', line)))
    except Exception:
        pass  # Ignore unreadable files
    return secrets

def scan_directory(root: str) -> List[Secret]:
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Ignore unwanted folders
        dirnames[:] = [d for d in dirnames if d not in IGNORE_FOLDERS]
        # Special handling for data/
        if os.path.basename(dirpath) == DATA_FOLDER:
            for fname in filenames:
                if is_allowed_file(fname):
                    fpath = os.path.join(dirpath, fname)
                    if os.path.getsize(fpath) <= MAX_DATA_FILE_SIZE:
                        results.extend(scan_file(fpath))
            dirnames.clear()  # Don't recurse into data/
            continue
        for fname in filenames:
            if is_allowed_file(fname):
                fpath = os.path.join(dirpath, fname)
                results.extend(scan_file(fpath))
    return results

def in_scan_scope(path: str) -> bool:
    """Whether the whole-repo scan would have looked at this file.

    Per-file mode follows the same scope, so pre-commit and CI agree: tests/
    (full of deliberate fake secrets) and root docs are not scanned.
    """
    norm = os.path.normpath(path).replace('\\', '/')
    if norm.startswith('./'):
        norm = norm[2:]
    parts = norm.split('/')
    if set(parts) & IGNORE_FOLDERS or '.git' in parts:
        return False
    if len(parts) == 1:
        return norm in SCAN_FILES
    return parts[0] in SCAN_FOLDERS or parts[0] == DATA_FOLDER


def scan_paths(paths: Iterable[str]) -> List[Secret]:
    """Scan just these files (what pre-commit passes: the staged files)."""
    results: List[Secret] = []
    for path in paths:
        if in_scan_scope(path) and os.path.isfile(path) and is_allowed_file(path):
            results.extend(scan_file(path))
    return results


def scan_repo() -> List[Secret]:
    all_secrets: List[Secret] = []
    for folder in SCAN_FOLDERS:
        if os.path.isdir(folder):
            all_secrets.extend(scan_directory(folder))
    for fname in SCAN_FILES:
        if os.path.isfile(fname) and is_allowed_file(fname):
            all_secrets.extend(scan_file(fname))
    # data/ at the root (config/ is already covered by SCAN_FOLDERS; scanning it
    # again here would duplicate results).
    if os.path.isdir(DATA_FOLDER):
        all_secrets.extend(scan_directory(DATA_FOLDER))
    return all_secrets


def load_baseline(path: Optional[str]) -> Set[str]:
    """Fingerprints of findings already reviewed and accepted."""
    if not path or not os.path.isfile(path):
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {str(x.get('fingerprint')) for x in data.get('accepted', []) if x.get('fingerprint')}
    except (OSError, ValueError, AttributeError):
        return set()


def write_baseline(path: str, secrets: List[Secret]) -> None:
    accepted = [{'fingerprint': s.fingerprint, 'file': s.file.replace('\\', '/'),
                 'type': s.type, 'confidence': s.confidence}
                for s in secrets if s.fingerprint]
    accepted.sort(key=lambda x: (x['file'], x['type'], x['fingerprint']))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'note': 'Reviewed findings accepted by audit_secrets.py --write-baseline.',
                   'accepted': accepted}, f, indent=2)
        f.write('\n')


def blocking(secrets: List[Secret], fail_on: str, baseline: Set[str]) -> List[Secret]:
    """Findings at or above ``fail_on`` that the baseline has not accepted."""
    if fail_on == 'none':
        return []
    threshold = CONFIDENCE_RANK[fail_on]
    return [s for s in secrets
            if CONFIDENCE_RANK.get(s.confidence, 0) >= threshold and s.fingerprint not in baseline]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Scan Hestia for committed secrets (backlog #197).",
        epilog="With no paths the whole repo is scanned. pre-commit passes the staged "
               "files; CI runs it with --fail-on high.")
    ap.add_argument('paths', nargs='*', help="files to scan instead of the whole repo")
    ap.add_argument('--fail-on', choices=['high', 'medium', 'low', 'none'], default='none',
                    help="exit 1 if a non-baselined finding is at least this confident "
                         "(default none: report only, exit 0, as before)")
    ap.add_argument('--baseline', default=DEFAULT_BASELINE,
                    help=f"accepted-findings file (default {DEFAULT_BASELINE})")
    ap.add_argument('--write-baseline', action='store_true',
                    help="accept everything found now into the baseline file, then exit 0")
    ap.add_argument('--no-report', action='store_true', help="do not write secrets_report.json")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    all_secrets = scan_paths(args.paths) if args.paths else scan_repo()

    if args.write_baseline:
        write_baseline(args.baseline, all_secrets)
        print(f"Wrote {len(all_secrets)} finding(s) to {args.baseline}.")
        return 0

    baseline = load_baseline(args.baseline)
    # Gate mode lists only what would fail the run, so CI output is actionable;
    # the default (report-only) mode still lists everything, as it always did.
    bad = blocking(all_secrets, args.fail_on, baseline)
    shown = bad if args.fail_on != 'none' else all_secrets

    if shown:
        print(f"{'File':60} {'Line':>5} {'Type':25} {'Conf':>6} {'Preview'}")
        print('-' * 120)
        for s in shown:
            print(f"{s.file:60} {s.line:5} {s.type:25} {s.confidence:6} {s.preview}")
    else:
        print("No secrets found.")

    if not args.no_report:
        report = [
            {'file': s.file, 'line': s.line, 'type': s.type,
             'confidence': s.confidence, 'preview': s.preview}
            for s in all_secrets
        ]
        with open('secrets_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

    if bad:
        print(f"\nFAILED: {len(bad)} finding(s) at or above '{args.fail_on}' confidence. "
              f"Remove the secret, mark a public-by-design line with '# {IGNORE_MARKER}', "
              f"or accept a reviewed false positive with --write-baseline.", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
