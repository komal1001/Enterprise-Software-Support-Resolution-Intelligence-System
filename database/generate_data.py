import random
from datetime import datetime, timedelta
import psycopg2

# ---------- CONFIG ----------
DB_CONFIG = {
    "dbname": "support_intelligence",
    "user": "postgres",
    "password": "",
    "host": "localhost",
    "port": 5432
}

NUM_CUSTOMERS = 120
NUM_TICKETS = 1000
NUM_INCIDENTS = 50
NUM_ARTICLES = 120

COMPANY_NAMES = [
    "Alpha Corp", "Beta Systems", "Gamma Retail", "Delta Logistics", "Epsilon Tech",
    "Zeta Dynamics", "Eta Solutions", "Theta Analytics", "Iota Ventures", "Kappa Industries",
    "Lambda Networks", "Mu Software", "Nu Consulting", "Xi Enterprises", "Omicron Labs",
    "Pi Technologies", "Rho Digital", "Sigma Group", "Tau Innovations", "Upsilon Media",
    "Phi Services", "Chi Robotics", "Psi Fintech", "Omega Cloud", "Apex Global",
    "Nexus Corp", "Vertex Systems", "Orion Tech", "Polaris Solutions", "Quantum Dynamics",
    "Stellar Networks", "Nova Analytics", "Pulsar Ventures", "Atlas Industries", "Titan Labs",
    "Helios Technologies", "Vega Digital", "Aquila Group", "Lyra Innovations", "Cygnus Media",
    "Draco Services", "Phoenix Robotics", "Hydra Fintech", "Pegasus Cloud", "Andromeda Global",
    "Cassiopeia Corp", "Perseus Systems", "Centaurus Tech", "Scorpius Solutions", "Gemini Dynamics",
    "Horizon Networks", "Meridian Analytics", "Zenith Ventures", "Pinnacle Industries", "Summit Labs",
    "Catalyst Technologies", "Synergy Digital", "Fusion Group", "Matrix Innovations", "Spectrum Media",
    "Prism Services", "Vortex Robotics", "Pulse Fintech", "Flux Cloud", "Core Global",
    "Vector Corp", "Proxy Systems", "Cipher Tech", "Mosaic Solutions", "Helix Dynamics",
    "Spiral Networks", "Orbit Analytics", "Comet Ventures", "Eclipse Industries", "Aurora Labs",
    "Solaris Technologies", "Lunar Digital", "Cosmic Group", "Galactic Innovations", "Nebula Media",
    "Asteroid Services", "Meteor Robotics", "Quasar Fintech", "Supernova Cloud", "Infinity Global",
    "Continuum Corp", "Paradigm Systems", "Blueprint Tech", "Archetype Solutions", "Prototype Dynamics",
    "Iterate Networks", "Agile Analytics", "Sprint Ventures", "Velocity Industries", "Momentum Labs",
    "Traction Technologies", "Leverage Digital", "Amplify Group", "Accelerate Innovations", "Scale Media",
    "Grow Services", "Expand Robotics", "Elevate Fintech", "Ascend Cloud", "Thrive Global",
    "Pioneer Corp", "Frontier Systems", "Vanguard Tech", "Avant Solutions", "Forward Dynamics",
    "Advance Networks", "Progress Analytics", "Evolve Ventures", "Transform Industries", "Innovate Labs",
    "Disrupt Technologies", "Redesign Digital", "Reimagine Group", "Reinvent Innovations", "Revamp Media",
    "Rebuild Services", "Reform Robotics", "Renew Fintech", "Refresh Cloud", "Revive Global",
]

random.seed(42)
# ----------------------------

def random_date(start_year=2023, end_year=2026):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def random_recent_datetime():
    now = datetime.now()
    return now - timedelta(days=random.randint(0, 180),
                           hours=random.randint(0, 23))

def determine_escalation(severity, sla):
    if severity == "Critical":
        return True
    if severity == "High" and sla == "Priority":
        return True
    return False

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    customer_ids = []

    # -------- Customers --------
    for i in range(NUM_CUSTOMERS):
        cur.execute("""
            INSERT INTO customers
            (company_name, subscription_tier, account_status,
             sla_level, renewal_date, region, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING customer_id
        """, (
            COMPANY_NAMES[i % len(COMPANY_NAMES)],
            random.choice(["Free", "Standard", "Premium", "Enterprise"]),
            random.choice(["Active", "Suspended", "Trial", "Cancelled"]),
            random.choice(["Basic", "Enhanced", "Priority"]),
            random_date(2025, 2026),
            random.choice(["US", "EU", "APAC", "MEA"]),
            random_date(2023, 2024)
        ))
        customer_ids.append(cur.fetchone()[0])

    # -------- Support Tickets --------
    for _ in range(NUM_TICKETS):
        customer_id = random.choice(customer_ids)
        severity = random.choice(["Low", "Medium", "High", "Critical"])
        sla = random.choice(["Basic", "Enhanced", "Priority"])
        escalation = determine_escalation(severity, sla)

        created = random_recent_datetime()
        resolved = None
        status = random.choice(["Open", "In Progress", "Resolved"])

        if status == "Resolved":
            resolved = created + timedelta(hours=random.randint(2, 72))

        cur.execute("""
            INSERT INTO support_tickets
            (customer_id, issue_category, severity_level,
             ticket_status, created_at, resolved_at,
             assigned_team, escalation_flag)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            customer_id,
            random.choice([
                "usage_configuration",
                "integration_api",
                "billing",
                "performance_latency",
                "security",
                "production_incident"
            ]),
            severity,
            status,
            created,
            resolved,
            random.choice(["L1", "L2", "Engineering", "Security"]),
            escalation
        ))

    # -------- Incident Logs --------
    for _ in range(NUM_INCIDENTS):
        start_time = random_recent_datetime()
        end_time = None
        resolution_status = random.choice(["Open", "Investigating", "Resolved"])

        if resolution_status == "Resolved":
            end_time = start_time + timedelta(hours=random.randint(1, 48))

        severity = random.choice(["Medium", "High", "Critical"])
        escalation_flag = severity == "Critical"

        cur.execute("""
            INSERT INTO incident_logs
            (incident_type, severity, affected_region,
             start_time, end_time, resolution_status,
             root_cause, escalation_flag)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            random.choice(["Outage", "API Failure", "Data Loss", "Security Alert"]),
            severity,
            random.choice(["US", "EU", "APAC", "MEA"]),
            start_time,
            end_time,
            resolution_status,
            "Synthetic generated incident root cause",
            escalation_flag
        ))

    # -------- Knowledge Articles --------
    for i in range(NUM_ARTICLES):
        cur.execute("""
            INSERT INTO knowledge_article_usage
            (article_title, product_version, category,
             last_updated, known_issue_flag, internal_confidence_score)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (
            f"Article_{i}_Troubleshooting_Guide",
            random.choice(["v1.0", "v2.1", "v3.0", "v3.2"]),
            random.choice(["Installation", "Configuration", "API", "Security", "Performance"]),
            random_date(2024, 2026),
            random.choice([True, False]),
            round(random.uniform(0.70, 0.99), 2)
        ))

    conn.commit()
    cur.close()
    conn.close()
    print("Enterprise Support dataset generated successfully.")

if __name__ == "__main__":
    main()
