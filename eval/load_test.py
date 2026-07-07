"""
Load test — Task 39 (Sprint 7)

Sends concurrent requests to POST /ticket and measures:
- P50, P95, P99 latency
- Success rate
- Rate limit triggers

Usage:
    python -m eval.load_test
"""

import asyncio
import time
import statistics
import httpx

API_URL     = "http://localhost:8000/ticket"
CONCURRENCY = 5   # simultaneous requests per batch
TOTAL       = 15  # total requests to send

TEST_TICKETS = [
    "How do I configure SSO for my organisation?",
    "Getting 401 Unauthorized when calling the REST API after rotating our key.",
    "Dashboard is loading very slowly — taking over 30 seconds.",
    "What are the webhook retry settings?",
    "How do I set up IP allowlisting for our enterprise account?",
    "API calls are timing out intermittently under high load.",
    "How do I export audit logs for compliance?",
    "Getting HTTP 429 Too Many Requests from the API.",
    "What is the SLA for Critical incidents?",
    "How do I configure multi-factor authentication?",
    "Our integration stopped sending webhook events since yesterday.",
    "How do I upgrade our subscription plan?",
    "What encryption standards does the platform use?",
    "Can you explain the rate limiting policy for the API?",
    "How do I onboard a new team member with read-only access?",
]


async def send_ticket(client: httpx.AsyncClient, ticket_text: str, idx: int) -> dict:
    payload = {
        "ticket_text": ticket_text,
        "thread_id":   f"load-test-{idx}",
        "conversation_history": [],
    }
    start = time.monotonic()
    try:
        resp = await client.post(API_URL, json=payload, timeout=60.0)
        latency_ms = (time.monotonic() - start) * 1000
        return {
            "idx":        idx,
            "status":     resp.status_code,
            "latency_ms": latency_ms,
            "ok":         resp.status_code == 200,
        }
    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        return {
            "idx":        idx,
            "status":     0,
            "latency_ms": latency_ms,
            "ok":         False,
            "error":      str(e),
        }


async def run_load_test():
    results = []

    async with httpx.AsyncClient() as client:
        tickets = (TEST_TICKETS * 3)[:TOTAL]

        for batch_start in range(0, TOTAL, CONCURRENCY):
            batch = tickets[batch_start: batch_start + CONCURRENCY]
            tasks = [
                send_ticket(client, text, batch_start + i)
                for i, text in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            done = min(batch_start + CONCURRENCY, TOTAL)
            print(f"  Completed {done}/{TOTAL} requests...")

    return results


def print_report(results: list[dict]):
    latencies   = [r["latency_ms"] for r in results]
    successes   = [r for r in results if r["ok"]]
    failures    = [r for r in results if not r["ok"]]
    rate_limits = [r for r in results if r.get("status") == 429]

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[min(int(len(latencies_sorted) * 0.99), len(latencies_sorted) - 1)]

    p95_slo_standard    = 5000
    p95_slo_multiagent  = 10000

    print("\n" + "=" * 60)
    print("LOAD TEST REPORT — Task 39 (Sprint 7)")
    print("=" * 60)
    print(f"Total requests   : {len(results)}")
    print(f"Successful        : {len(successes)}")
    print(f"Failed            : {len(failures)}")
    print(f"Rate limited (429): {len(rate_limits)}")
    print(f"Success rate      : {len(successes) / len(results) * 100:.1f}%")
    print()
    print("--- Latency ---")
    print(f"  Min  : {min(latencies):>8.0f} ms")
    print(f"  P50  : {p50:>8.0f} ms")
    print(f"  P95  : {p95:>8.0f} ms  (SLO standard <= {p95_slo_standard}ms)  {'PASS' if p95 <= p95_slo_standard else 'FAIL'}")
    print(f"  P99  : {p99:>8.0f} ms  (SLO multi-agent <= {p95_slo_multiagent}ms)  {'PASS' if p99 <= p95_slo_multiagent else 'FAIL'}")
    print(f"  Max  : {max(latencies):>8.0f} ms")
    print()

    if failures:
        print("--- Failures ---")
        for r in failures:
            err = r.get("error", f"HTTP {r['status']}")
            print(f"  [{r['idx']:>2}] {err[:80]}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    print(f"Starting load test — {TOTAL} requests, {CONCURRENCY} concurrent")
    print(f"Target: {API_URL}\n")
    results = asyncio.run(run_load_test())
    print_report(results)
