"""Safe, offline, deterministic string-level URL perturbations.

Public API:  perturb(url, family, severity, seed) -> str

Families (targeted):
    path_padding       severity = number of benign path tokens appended
    subdomain_padding  severity = number of benign labels prepended to host
    char_edit          severity = fraction (0-1) of eligible chars edited
                       (digit look-alike swaps, hyphen insertion)
Controls (same length change, random lowercase letters instead):
    control_path, control_subdomain, control_char

Nothing here resolves, fetches or generates working URLs. Strings only.
"""
import hashlib
import random
import re
import string
from urllib.parse import urlsplit, urlunsplit

PATH_TOKENS = ["docs", "help", "about", "index", "support", "news",
               "contact", "static", "assets", "blog", "faq", "home"]
SUBDOMAIN_TOKENS = ["www", "mail", "secure", "support", "portal", "app", "my", "web"]
LOOKALIKE = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}

TARGETED = [
    "path_padding",
    "subdomain_padding",
    "char_edit",
    "path_strip",
]
CONTROL_OF = {"control_path": "path_padding",
              "control_subdomain": "subdomain_padding",
              "control_char": "char_edit"}
ALL_FAMILIES = TARGETED + list(CONTROL_OF)

_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _rng(seed, url, base_family, severity):
    # Controls share the base family's key so they make the same choices.
    key = f"{seed}|{url}|{base_family}|{severity!r}".encode()
    return random.Random(int(hashlib.sha256(key).hexdigest(), 16))


def _rand_letters(lrng, n):
    return "".join(lrng.choice(string.ascii_lowercase) for _ in range(n))


def _split_netloc(netloc):
    """-> (userinfo_prefix, host, port_suffix). Keeps '@' and ':' pieces."""
    prefix = ""
    if "@" in netloc:
        userinfo, _, netloc = netloc.rpartition("@")
        prefix = userinfo + "@"
    port = ""
    if not netloc.startswith("[") and ":" in netloc:
        host, _, p = netloc.rpartition(":")
        if p.isdigit():
            netloc, port = host, ":" + p
    return prefix, netloc, port


def _is_ip(host):
    return bool(_IPV4.match(host)) or host.startswith("[")


def _parse(url):
    """Return (parts, had_scheme). Adds a temporary scheme if missing."""
    had_scheme = "://" in url
    parts = urlsplit(url if had_scheme else "http://" + url)
    return parts, had_scheme


def _rebuild(parts, had_scheme):
    out = urlunsplit(parts)
    return out if had_scheme else out.split("://", 1)[1]


def _path_padding(url, n, rng, control, lrng):
    parts, hs = _parse(url)
    if not parts.netloc:
        return url
    tokens = [rng.choice(PATH_TOKENS) for _ in range(int(n))]
    if control:  # same lengths, random letters
        tokens = [_rand_letters(lrng, len(t)) for t in tokens]
    path = parts.path.rstrip("/") + "/" + "/".join(tokens)
    return _rebuild(parts._replace(path=path), hs)


def _subdomain_padding(url, n, rng, control, lrng):
    parts, hs = _parse(url)
    prefix, host, port = _split_netloc(parts.netloc)
    if not host or _is_ip(host):
        return url
    labels = [rng.choice(SUBDOMAIN_TOKENS) for _ in range(int(n))]
    if control:
        labels = [_rand_letters(lrng, len(t)) for t in labels]
    new_host = ".".join(labels + [host])
    return _rebuild(parts._replace(netloc=prefix + new_host + port), hs)


def _edit_segment(s, frac, rng, control, lrng):
    """Edit ~frac of alphanumeric chars of s. Length change identical for
    targeted and control (look-alike -> swap, otherwise insert one char)."""
    eligible = [i for i, c in enumerate(s) if c.isascii() and c.isalnum()]
    if not eligible:
        return s
    k = min(len(eligible), max(1, round(frac * len(eligible))))
    chosen = set(rng.sample(eligible, k))
    out = []
    for i, c in enumerate(s):
        if i not in chosen:
            out.append(c)
            continue
        if c.lower() in LOOKALIKE:
            out.append(lrng.choice(string.ascii_lowercase) if control
                       else LOOKALIKE[c.lower()])
        elif i + 1 < len(s) and s[i + 1].isalnum():
            out.append(c + (lrng.choice(string.ascii_lowercase) if control else "-"))
        else:
            out.append(c)
    return "".join(out)


def _char_edit(url, frac, rng, control, lrng):
    parts, hs = _parse(url)
    prefix, host, port = _split_netloc(parts.netloc)
    if not host:
        return url
    new_host = host
    if not _is_ip(host) and "." in host:
        body, _, tld = host.rpartition(".")  # never touch the TLD
        new_host = _edit_segment(body, frac, rng, control, lrng) + "." + tld
    new_path = _edit_segment(parts.path, frac, rng, control, lrng)
    return _rebuild(parts._replace(netloc=prefix + new_host + port,
                                   path=new_path), hs)


def perturb(url: str, family: str, severity, seed: int = 42) -> str:
    """Deterministic: same (url, family, severity, seed) -> same output."""
    if family not in ALL_FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {ALL_FAMILIES}")
    control = family in CONTROL_OF
    base = CONTROL_OF.get(family, family)
    rng = _rng(seed, url, base, severity)
    lrng = _rng(seed, url, base + "#letters", severity)  # separate stream
    fn = {"path_padding": _path_padding,
          "subdomain_padding": _subdomain_padding,
          "char_edit": _char_edit,
          "path_strip": _path_strip,}[base]
    return fn(url, severity, rng, control, lrng)

def _path_strip(url, level, rng, control, lrng):
    parts, hs = _parse(url)

    if not parts.netloc:
        return url

    segs = [s for s in parts.path.split("/") if s]
    level = int(level)

    if level == 1:
        new = parts._replace(query="", fragment="")
    elif level == 2:
        new = parts._replace(
            path=("/" + segs[0]) if segs else "",
            query="",
            fragment=""
        )
    else:
        new = parts._replace(
            path="",
            query="",
            fragment=""
        )

    return _rebuild(new, hs)