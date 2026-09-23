"""
generate_data.py - Realistic synthetic network-flow generator
=============================================================

Purpose
-------
Lets the whole project run "out of the box" without downloading multi-gigabyte
benchmark datasets (CICIDS2017, NSL-KDD ...). The generator emits *raw,
un-preprocessed* flow records with the same kind of statistical pathologies real
NetFlow / CICFlowMeter data has:

* heavy-tailed (log-normal) durations and byte rates  -> needs log(1+x)
* wildly different feature scales (packets vs. microseconds) -> needs scaling
* identifier columns (IP, port, timestamp) that must NOT be fed to a model

Traffic mix (default 95 % benign / 5 % malicious)
-------------------------------------------------
Benign profiles (mixture model - real networks are multi-modal, which is exactly
why density-based detectors are interesting):
    web   : short HTTP(S) request/response flows
    dns   : tiny UDP query/response pairs
    bulk  : long downloads / streaming, many large packets
    ssh   : interactive sessions, many small packets, moderate duration

Malicious profiles:
    PortScan     : micro-flows, 1-3 SYNs, (almost) no reply, random dst ports
    DoS          : SYN-flood style bursts, thousands of small packets, no reply
    Exfiltration : very long flows, huge one-directional upload, few replies

Ground truth is written to `label` (0/1) and `attack_type` so that the
evaluator can score the *unsupervised* models. Neither column is ever shown to
the models during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED = 42          # fixed seed -> reproducible datasets (important for demos/tests)
TCP, UDP = 6, 17           # IANA protocol numbers


def _ips(rng: np.random.Generator, n: int, prefix: str, n_hosts: int) -> np.ndarray:
    """Draw `n` IPv4 strings of the form '<prefix>.<host>' with host in [1, n_hosts].

    Restricting the host pool models the fact that a small number of machines
    generate most of the traffic (a handful of servers, a few hundred clients).
    """
    hosts = rng.integers(1, n_hosts + 1, size=n)
    return np.char.add(f"{prefix}.", hosts.astype(str))


def _assemble(rng, duration_us, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes,
              syn, ack, protocol, dst_port, src_ip, dst_ip, attack_type) -> pd.DataFrame:
    """Turn the per-flow base quantities into a full flow-record DataFrame.

    Only *base* quantities are simulated (duration, packet counts, byte counts,
    flag counts). Derived rates (bytes/s, packets/s, average packet size) are
    then *computed* from them exactly as CICFlowMeter does, so the columns are
    internally consistent - a detector cannot cheat on inconsistent columns and
    real-world correlations between features are preserved.
    """
    n = len(duration_us)
    # Durations are in MICROSECONDS (CICIDS2017 convention). >= 1 avoids divide-by-zero.
    duration_us = np.maximum(np.round(duration_us), 1).astype(np.int64)
    fwd_pkts = np.maximum(np.round(fwd_pkts), 1).astype(np.int64)   # every flow has >=1 forward packet
    bwd_pkts = np.maximum(np.round(bwd_pkts), 0).astype(np.int64)   # replies may be absent (scans, floods)
    fwd_bytes = np.maximum(np.round(fwd_bytes), 1).astype(np.int64)
    bwd_bytes = np.maximum(np.round(bwd_bytes), 0).astype(np.int64)
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes
    seconds = duration_us / 1e6                      # convert to seconds for "per second" rates

    return pd.DataFrame({
        # ---- identifier columns: dropped by the preprocessor (would let a model memorise hosts)
        "src_ip": src_ip,
        "src_port": rng.integers(49152, 65536, size=n),   # ephemeral client port range
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "protocol": protocol,
        # ---- behavioural features: what the models actually see
        "flow_duration": duration_us,
        "total_fwd_packets": fwd_pkts,
        "total_bwd_packets": bwd_pkts,
        "total_fwd_bytes": fwd_bytes,
        "total_bwd_bytes": bwd_bytes,
        "flow_bytes_s": np.round(total_bytes / seconds, 3),      # bytes per second
        "flow_packets_s": np.round(total_pkts / seconds, 3),     # packets per second
        "avg_packet_size": np.round(total_bytes / total_pkts, 3),
        "syn_flag_count": np.maximum(np.round(syn), 0).astype(np.int64),
        "ack_flag_count": np.clip(np.round(ack), 0, total_pkts).astype(np.int64),  # cannot exceed #packets
        # ---- ground truth (evaluation only)
        "attack_type": attack_type,
    })


# --------------------------------------------------------------------------- #
# Benign traffic profiles
# --------------------------------------------------------------------------- #
def _benign(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate `n` benign flows from a 4-component mixture."""
    # Mixture weights: web-dominated, plenty of DNS, some bulk transfers, little SSH.
    kinds = rng.choice(["web", "dns", "bulk", "ssh"], size=n, p=[0.55, 0.20, 0.15, 0.10])
    frames = []
    for kind in ("web", "dns", "bulk", "ssh"):
        m = int((kinds == kind).sum())
        if m == 0:
            continue
        if kind == "web":
            # Log-normal duration: the canonical model for network flow lengths (multiplicative effects).
            dur = rng.lognormal(np.log(150_000), 1.0, m)               # median 150 ms
            fwd, bwd = rng.poisson(10, m) + 1, rng.poisson(16, m) + 1  # Poisson packet counts
            fb = fwd * np.clip(rng.normal(250, 80, m), 40, 1500)       # small requests
            bb = bwd * np.clip(rng.normal(900, 300, m), 40, 1500)      # larger responses
            syn = np.where(rng.random(m) < 0.05, 2, 1)                 # one handshake (occasional retry)
            ack = fwd + bwd - rng.integers(0, 2, m)                    # nearly every packet carries ACK
            proto, port = np.full(m, TCP), rng.choice([80, 443, 8080, 8443], m, p=[.2, .7, .05, .05])
            dst = _ips(rng, m, "172.16.0", 30)                         # ~30 web servers
        elif kind == "dns":
            dur = rng.lognormal(np.log(30_000), 0.6, m)
            fwd, bwd = rng.integers(1, 3, m), rng.integers(1, 3, m)
            fb, bb = fwd * rng.normal(75, 10, m), bwd * rng.normal(140, 40, m)
            syn, ack = np.zeros(m), np.zeros(m)                        # UDP: no TCP flags at all
            proto, port = np.full(m, UDP), np.full(m, 53)
            dst = np.full(m, "10.0.0.2")                               # single internal resolver
        elif kind == "bulk":
            dur = rng.lognormal(np.log(4_000_000), 0.8, m)             # median 4 s
            fwd, bwd = rng.poisson(120, m) + 10, rng.poisson(260, m) + 10
            fb = fwd * np.clip(rng.normal(300, 100, m), 40, 1500)
            bb = bwd * np.clip(rng.normal(1200, 150, m), 40, 1500)     # big downstream packets
            syn = np.ones(m)
            ack = fwd + bwd - rng.integers(0, 3, m)
            proto, port = np.full(m, TCP), rng.choice([443, 21, 445], m, p=[.7, .1, .2])
            dst = _ips(rng, m, "172.16.1", 10)
        else:  # ssh
            dur = rng.lognormal(np.log(2_500_000), 1.0, m)
            fwd, bwd = rng.poisson(45, m) + 2, rng.poisson(45, m) + 2
            fb = fwd * np.clip(rng.normal(120, 40, m), 40, 1500)       # keystrokes: small packets
            bb = bwd * np.clip(rng.normal(160, 60, m), 40, 1500)
            syn = np.ones(m)
            ack = fwd + bwd - rng.integers(0, 3, m)
            proto, port = np.full(m, TCP), np.full(m, 22)
            dst = _ips(rng, m, "10.0.2", 15)
        src = _ips(rng, m, "10.0.1", 200)                              # ~200 internal clients
        frames.append(_assemble(rng, dur, fwd, bwd, fb, bb, syn, ack, proto, port,
                                src, dst, np.full(m, "BENIGN")))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Malicious traffic profiles
# --------------------------------------------------------------------------- #
def _port_scan(rng, m):
    """Horizontal/vertical scan: one or two SYN probes, RST or silence, random ports."""
    dur = rng.uniform(50, 5_000, m)                          # microseconds: probes end almost instantly
    fwd, bwd = rng.integers(1, 4, m), rng.integers(0, 2, m)
    fb, bb = fwd * rng.normal(44, 4, m), bwd * rng.normal(40, 4, m)   # bare TCP/IP header sized packets
    syn = fwd                                                # every forward packet is a bare SYN
    ack = rng.integers(0, 2, m) * bwd                        # RST/ACK replies only from closed ports
    port = rng.integers(1, 65536, m)                         # sweeping the whole port space
    return _assemble(rng, dur, fwd, bwd, fb, bb, syn, ack, np.full(m, TCP), port,
                     _ips(rng, m, "198.51.100", 5),          # only a few external scanners
                     _ips(rng, m, "10.0.1", 200), np.full(m, "PortScan"))


def _dos(rng, m):
    """SYN-flood style burst: a torrent of tiny packets, almost no answers."""
    dur = rng.lognormal(np.log(800_000), 0.7, m)
    fwd = rng.lognormal(np.log(6_000), 0.7, m)               # thousands of packets per flow
    bwd = rng.integers(0, 30, m)
    fb, bb = fwd * rng.normal(60, 10, m), bwd * rng.normal(60, 10, m)
    syn = fwd * rng.uniform(0.7, 0.95, m)                    # flood is mostly SYNs...
    ack = fwd * rng.uniform(0.0, 0.15, m)                    # ...with few completed handshakes
    port = rng.choice([80, 443], m)
    return _assemble(rng, dur, fwd, bwd, fb, bb, syn, ack, np.full(m, TCP), port,
                     _ips(rng, m, "203.0.113", 40),          # botnet of ~40 sources
                     np.full(m, "172.16.0.10"), np.full(m, "DoS"))   # one victim server


def _exfil(rng, m):
    """Data exfiltration: long, heavily asymmetric upload to an external host."""
    dur = rng.lognormal(np.log(60_000_000), 0.5, m)          # median 60 s
    fwd = rng.lognormal(np.log(15_000), 0.5, m)              # huge upload
    bwd = fwd * rng.uniform(0.02, 0.08, m)                   # only sparse ACK-like replies
    fb, bb = fwd * np.clip(rng.normal(1300, 100, m), 40, 1500), bwd * rng.normal(60, 10, m)
    syn, ack = np.ones(m), fwd + bwd
    port = rng.choice([443, 53, 8443, 22], m, p=[.5, .2, .2, .1])   # HTTPS / DNS-tunnel / SSH
    return _assemble(rng, dur, fwd, bwd, fb, bb, syn, ack, np.full(m, TCP), port,
                     _ips(rng, m, "10.0.1", 200),            # compromised internal host
                     _ips(rng, m, "192.0.2", 3), np.full(m, "Exfiltration"))


def generate_synthetic_traffic(n_flows: int = 20_000, anomaly_ratio: float = 0.05,
                               seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return a shuffled DataFrame of `n_flows` flows, `anomaly_ratio` of them malicious.

    Malicious flows are split 40 % PortScan / 35 % DoS / 25 % Exfiltration.
    """
    if not 0.0 <= anomaly_ratio < 1.0:
        raise ValueError("anomaly_ratio must be in [0, 1)")
    rng = np.random.default_rng(seed)
    n_bad = int(round(n_flows * anomaly_ratio))
    n_scan, n_dos = int(round(n_bad * 0.40)), int(round(n_bad * 0.35))
    n_exfil = n_bad - n_scan - n_dos
    parts = [_benign(rng, n_flows - n_bad)]
    for fn, k in ((_port_scan, n_scan), (_dos, n_dos), (_exfil, n_exfil)):
        if k > 0:
            parts.append(fn(rng, k))
    df = pd.concat(parts, ignore_index=True)

    # Shuffle so attacks are interleaved with benign flows (no ordering leakage), then
    # assign chronologically sorted timestamps across one simulated day.
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    seconds = np.sort(rng.uniform(0, 86_400, len(df)))
    stamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(seconds, unit="s")
    df.insert(0, "timestamp", stamps.strftime("%Y-%m-%d %H:%M:%S").to_numpy())

    # Binary ground truth: 0 = benign, 1 = any attack.
    df["label"] = (df["attack_type"] != "BENIGN").astype(int)
    return df


def main() -> None:
    """CLI: python -m src.generate_data --n-flows 20000 --output data/synthetic_traffic.csv"""
    p = argparse.ArgumentParser(description="Generate synthetic network-flow data.")
    p.add_argument("--n-flows", type=int, default=20_000)
    p.add_argument("--anomaly-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=str, default="data/synthetic_traffic.csv")
    a = p.parse_args()
    df = generate_synthetic_traffic(a.n_flows, a.anomaly_ratio, a.seed)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} flows ({int(df['label'].sum()):,} malicious) -> {out}")


if __name__ == "__main__":
    main()
