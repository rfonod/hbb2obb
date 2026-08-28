#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import io
import json

import pytest

import hbb2obb.version_check as vc
from hbb2obb import __version__


@pytest.fixture
def cache_in_tmp(tmp_path, monkeypatch):
    """Isolate the update-check cache and the once-per-process guard for each test."""
    monkeypatch.setattr(vc, '_cache_dir', lambda: tmp_path)
    vc._state['checked'] = False
    monkeypatch.delenv(vc.ENV_OPT_OUT, raising=False)
    yield tmp_path
    vc._state['checked'] = False


def _fake_urlopen(version):
    def _open(url, timeout=None):
        payload = json.dumps({'info': {'version': version}}).encode()
        return io.BytesIO(payload)

    return _open


def test_newer_version_warns(cache_in_tmp, monkeypatch, capsys):
    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen('999.0.0'))
    vc.check_for_updates(blocking=True)
    assert '999.0.0' in capsys.readouterr().err


def test_same_version_silent(cache_in_tmp, monkeypatch, capsys):
    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen(__version__))
    vc.check_for_updates(blocking=True)
    assert capsys.readouterr().err == ""


def test_network_error_silent(cache_in_tmp, monkeypatch, capsys):
    def _raise(url, timeout=None):
        raise OSError('offline')

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _raise)
    vc.check_for_updates(blocking=True)
    assert capsys.readouterr().err == ""


def test_bad_json_silent(cache_in_tmp, monkeypatch, capsys):
    monkeypatch.setattr(vc.urllib.request, 'urlopen', lambda url, timeout=None: io.BytesIO(b'not json'))
    vc.check_for_updates(blocking=True)
    assert capsys.readouterr().err == ""


def test_opt_out_skips_network(cache_in_tmp, monkeypatch):
    called = []

    def _open(url, timeout=None):
        called.append(url)
        raise AssertionError('should not be called')

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _open)
    monkeypatch.setenv(vc.ENV_OPT_OUT, '1')
    vc.check_for_updates(blocking=True)
    assert called == []


def test_fresh_cache_skips_network(cache_in_tmp, monkeypatch, capsys):
    vc._write_cache('999.0.0')
    called = []

    def _open(url, timeout=None):
        called.append(url)
        raise AssertionError('should not be called')

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _open)
    vc.check_for_updates(blocking=True)
    assert called == []
    assert '999.0.0' in capsys.readouterr().err


def test_stale_cache_fetches(cache_in_tmp, monkeypatch, capsys):
    (cache_in_tmp / 'update_check.json').write_text(json.dumps({'last_check': 0, 'latest_version': '0.0.1'}))
    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen('999.0.0'))
    vc.check_for_updates(blocking=True)
    assert '999.0.0' in capsys.readouterr().err


def test_logger_used_when_provided(cache_in_tmp, monkeypatch, capsys):
    """A supplied logger takes precedence over the stderr fallback."""
    messages = []

    class _Logger:
        def warning(self, message):
            messages.append(message)

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen('999.0.0'))
    vc.check_for_updates(_Logger(), blocking=True)
    assert any('999.0.0' in m for m in messages)
    assert capsys.readouterr().err == ""


def test_check_once(cache_in_tmp, monkeypatch):
    calls = []
    monkeypatch.setattr(vc, 'check_for_updates', lambda logger=None: calls.append(1))
    vc.check_for_updates_once()
    vc.check_for_updates_once()
    assert len(calls) == 1


def test_parse_version_never_raises():
    assert vc._parse_version('1.4.0') == (1, 4, 0)
    assert vc._parse_version('1.4.0.dev1') == (1, 4, 0, 0)
    assert vc._parse_version('2.0.0rc1') == (2, 0, 0)
    assert vc._parse_version('') == (0,)
