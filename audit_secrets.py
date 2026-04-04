import os
import re
import json
import base64
import yaml
import hashlib
from collections import namedtuple
from typing import List, Dict, Any

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

Secret = namedtuple('Secret', ['file', 'line', 'type', 'confidence', 'preview'])

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
    entropy = 0
    for x in set(data):
        p_x = float(data.count(x)) / len(data)
        entropy -= p_x * (p_x and (p_x).bit_length())
    return entropy

def scan_file(filepath: str) -> List[Secret]:
    secrets = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                for pattern, secret_type, confidence in PATTERNS:
                    for match in pattern.finditer(line):
                        key = match.group(1) if match.lastindex and match.lastindex >= 1 else ''
                        value = match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(0)
                        if is_allowlisted(key, value):
                            continue
                        preview = line.strip()[:50]
                        secrets.append(Secret(filepath, i, secret_type, confidence, preview))
                # Entropy-based detection for unknown secrets
                for word in re.findall(r'([A-Za-z0-9+/=]{20,})', line):
                    if len(word) > 20 and shannon_entropy(word) > 4.0 and not is_allowlisted('', word):
                        preview = line.strip()[:50]
                        secrets.append(Secret(filepath, i, 'High-Entropy String', 'low', preview))
    except Exception as e:
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

def main():
    all_secrets = []
    # Scan folders
    for folder in SCAN_FOLDERS:
        if os.path.isdir(folder):
            all_secrets.extend(scan_directory(folder))
    # Scan top-level files
    for fname in SCAN_FILES:
        if os.path.isfile(fname) and is_allowed_file(fname):
            all_secrets.extend(scan_file(fname))
    # Scan config/ and data/ at root if present
    for special in ['config', DATA_FOLDER]:
        if os.path.isdir(special):
            all_secrets.extend(scan_directory(special))
    # Output to console (table)
    if all_secrets:
        print(f"{'File':60} {'Line':>5} {'Type':25} {'Conf':>6} {'Preview'}")
        print('-' * 120)
        for s in all_secrets:
            print(f"{s.file:60} {s.line:5} {s.type:25} {s.confidence:6} {s.preview}")
    else:
        print("No secrets found.")
    # Output to JSON
    report = [
        {
            'file': s.file,
            'line': s.line,
            'type': s.type,
            'confidence': s.confidence,
            'preview': s.preview
        }
        for s in all_secrets
    ]
    with open('secrets_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()