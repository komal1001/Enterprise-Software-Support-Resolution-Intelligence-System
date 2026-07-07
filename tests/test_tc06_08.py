from src.agents.agent1_classify import agent1_classify

cases = [
    ("TC-06", "Our API response times have degraded significantly over the past 24 hours - requests that used to complete in under 200ms are now taking 3 to 5 seconds and affecting our production application."),
    ("TC-08", "Our webhook endpoint stopped receiving event notifications 3 days ago. The webhook URL is configured correctly in our settings but no events are being delivered."),
]

for tc, text in cases:
    state = {
        "ticket_text": text,
        "conversation_history": [],
        "classification": None,
        "rag_result": None,
        "sql_result": None,
        "severity_assessment": None,
        "escalation_package": None,
        "final_response": None,
        "reflection_count": 0,
    }
    try:
        result_state = agent1_classify(state)
        r = result_state.get("classification")
        if r:
            print(f"{tc}: {r['category']} / {r['severity']} / {r['routing_path']} (conf={r['confidence']})")
        else:
            print(f"{tc}: classification returned None — full state: {result_state}")
    except Exception as e:
        print(f"{tc}: EXCEPTION — {type(e).__name__}: {e}")
