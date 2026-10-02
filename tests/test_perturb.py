import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perturb import ALL_FAMILIES, CONTROL_OF, perturb  # noqa: E402

URLS = [
    "https://www.example.com/",
    "http://login-secure.bank.co.uk/account/verify?id=123",
    "https://sub.domain.example.org:8080/a/b/c.html",
    "http://192.168.1.10/index.php",
    "http://example.com",
    "https://example.com/path/to/page?x=1&y=2#frag",
    "http://user@evil.example.net/login",
    "example.com/no-scheme",
    "https://a.b",
    "http://xn--80ak6aa92e.com/",
]
SEV = {
    "path_padding": [1, 3, 6],
    "subdomain_padding": [1, 2, 4],
    "char_edit": [0.10, 0.20, 0.40],
    "path_strip": [1, 2, 3],
}


def sev_for(fam):
    return SEV[CONTROL_OF.get(fam, fam)]


@pytest.mark.parametrize("fam", ALL_FAMILIES)
@pytest.mark.parametrize("url", URLS)
def test_no_crash_and_parseable(fam, url):
    for s in sev_for(fam):
        out = perturb(url, fam, s, 42)
        assert isinstance(out, str) and out
        urlsplit(out if "://" in out else "http://" + out)


@pytest.mark.parametrize("fam", ALL_FAMILIES)
def test_deterministic(fam):
    for url in URLS:
        s = sev_for(fam)[1]
        assert perturb(url, fam, s, 42) == perturb(url, fam, s, 42)


def test_seed_changes_output():
    u = "https://www.example.com/a/b"
    assert perturb(u, "path_padding", 6, 1) != perturb(u, "path_padding", 6, 2)


def test_path_padding_grows_with_severity():
    u = "https://www.example.com/a"
    lens = [len(perturb(u, "path_padding", s, 42)) for s in (1, 3, 6)]
    assert lens[0] < lens[1] < lens[2]


def test_subdomain_padding_and_ip_skip():
    out = perturb("https://example.com/", "subdomain_padding", 2, 42)
    assert urlsplit(out).hostname.endswith(".example.com")
    assert len(urlsplit(out).hostname.split(".")) == 4
    ip = "http://192.168.1.10/index.php"
    assert perturb(ip, "subdomain_padding", 4, 42) == ip


def test_char_edit_keeps_scheme_and_tld():
    u = "https://secure-login.paypal-service.com/account/verify"
    out = perturb(u, "char_edit", 0.4, 42)
    assert out != u
    assert out.startswith("https://") and urlsplit(out).hostname.endswith(".com")


@pytest.mark.parametrize("ctrl", list(CONTROL_OF))
def test_control_length_matches_targeted(ctrl):
    base = CONTROL_OF[ctrl]
    for url in URLS:
        for s in SEV[base]:
            assert len(perturb(url, ctrl, s, 42)) == len(perturb(url, base, s, 42))


def test_unknown_family():
    with pytest.raises(ValueError):
        perturb("http://a.com", "nope", 1, 42)