import os
import uuid
import datetime
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Fraud Detection Engine (MOCK)")

# --- IN-MEMORY DATABASE ---

# Mock Transactions list
# Each transaction fields map to the Kotlin Entity:
# id, externalId, accountId, amount, currency, merchant, merchantCategory, location, ipAddress, deviceId, channel, status, createdAt, processedAt
TRANSACTIONS = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "externalId": "ext_txn_001",
        "accountId": "acc_mario",
        "amount": 45.50,
        "currency": "EUR",
        "merchant": "Sainsbury's",
        "merchantCategory": "GROCERY",
        "location": "London, UK",
        "ipAddress": "192.168.1.50",
        "deviceId": "dev_macbook_001",
        "channel": "web",
        "status": "ALLOWED",
        "createdAt": (datetime.datetime.now() - datetime.timedelta(hours=5)).isoformat(),
        "processedAt": (datetime.datetime.now() - datetime.timedelta(hours=5)).isoformat()
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "externalId": "ext_txn_002",
        "accountId": "acc_mario",
        "amount": 1250.00,
        "currency": "EUR",
        "merchant": "Apple Store",
        "merchantCategory": "ELECTRONICS",
        "location": "London, UK",
        "ipAddress": "192.168.1.50",
        "deviceId": "dev_macbook_001",
        "channel": "web",
        "status": "REVIEW",
        "createdAt": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat(),
        "processedAt": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat()
    },
    {
        "id": "33333333-3333-3333-3333-333333333333",
        "externalId": "ext_txn_003",
        "accountId": "acc_alice",
        "amount": 5400.00,
        "currency": "USD",
        "merchant": "Crypto Exchange",
        "merchantCategory": "FINANCIAL",
        "location": "Unknown",
        "ipAddress": "45.12.99.10",
        "deviceId": "dev_android_999",
        "channel": "mobile",
        "status": "BLOCKED",
        "createdAt": (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat(),
        "processedAt": (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
    }
]

# Mock Alerts list
# id, transactionId, accountId, modelVersion, fraudScore, thresholdUsed, featuresUsed, decision, resolved, resolvedBy, resolvedAt, createdAt
ALERTS = [
    {
        "id": "a1111111-1111-1111-1111-111111111111",
        "transactionId": "22222222-2222-2222-2222-222222222222",
        "accountId": "acc_mario",
        "modelVersion": "isolation_forest_v1.2.0",
        "fraudScore": 0.7200,
        "thresholdUsed": 0.6500,
        "featuresUsed": {
            "amount": 1250.00,
            "hour_of_day": 14,
            "day_of_week": 1,
            "txn_per_hour": 3,
            "distinct_merchants": 2,
            "avg_txn_amount": 150.00,
            "is_foreign": 0,
            "is_new_device": 0
        },
        "decision": "REVIEW",
        "resolved": False,
        "resolvedBy": None,
        "resolvedAt": None,
        "createdAt": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat()
    },
    {
        "id": "a2222222-2222-2222-2222-222222222222",
        "transactionId": "33333333-3333-3333-3333-333333333333",
        "accountId": "acc_alice",
        "modelVersion": "isolation_forest_v1.2.0",
        "fraudScore": 0.9150,
        "thresholdUsed": 0.6500,
        "featuresUsed": {
            "amount": 5400.00,
            "hour_of_day": 15,
            "day_of_week": 1,
            "txn_per_hour": 1,
            "distinct_merchants": 1,
            "avg_txn_amount": 120.00,
            "is_foreign": 1,
            "is_new_device": 1
        },
        "decision": "BLOCK",
        "resolved": False,
        "resolvedBy": None,
        "resolvedAt": None,
        "createdAt": (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
    }
]

# Mock Audit Logs
# id, eventType, actor, entityType, entityId, payload, createdAt
AUDIT_LOGS = [
    {
        "id": 1,
        "eventType": "TRANSACTION_FLAGGED_FOR_REVIEW",
        "actor": "fraud-detection-service",
        "entityType": "transaction",
        "entityId": "22222222-2222-2222-2222-222222222222",
        "payload": {
            "decision": "REVIEW",
            "score": 0.7200,
            "reason": "Transaction amount significantly exceeds average historical transaction amount."
        },
        "createdAt": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat()
    },
    {
        "id": 2,
        "eventType": "TRANSACTION_BLOCKED",
        "actor": "fraud-detection-service",
        "entityType": "transaction",
        "entityId": "33333333-3333-3333-3333-333333333333",
        "payload": {
            "decision": "BLOCK",
            "score": 0.9150,
            "reason": "Foreign transaction from unseen device exceeding risk thresholds."
        },
        "createdAt": (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
    }
]


# ─── ALERT TOOLS ──────────────────────────────────────────────────────────────

@mcp.tool()
async def get_unresolved_alerts() -> Any:
    """
    Fetch all unresolved alerts (the analyst queue) that require manual review.
    """
    return [alert for alert in ALERTS if not alert["resolved"]]

@mcp.tool()
async def get_alerts_by_account(account_id: str) -> Any:
    """
    Fetch all fraud alerts for a specific account.
    """
    return [alert for alert in ALERTS if alert["accountId"] == account_id]

@mcp.tool()
async def get_alerts_by_decision(decision: str) -> Any:
    """
    Fetch alerts by decision type (ALLOW, REVIEW, BLOCK).
    """
    upper_decision = decision.upper()
    if upper_decision not in ("ALLOW", "REVIEW", "BLOCK"):
        return {"error": "Invalid Decision", "message": "Decision must be one of: ALLOW, REVIEW, BLOCK"}
    return [alert for alert in ALERTS if alert["decision"] == upper_decision]

@mcp.tool()
async def get_alert_by_id(alert_id: str) -> Any:
    """
    Fetch detailed info for a single fraud alert by its UUID.
    """
    for alert in ALERTS:
        if alert["id"] == alert_id:
            return alert
    return {"error": "Not Found", "message": f"Alert not found for ID: {alert_id}"}

@mcp.tool()
async def resolve_alert(alert_id: str, resolved_by: str) -> Any:
    """
    Mark a fraud alert as resolved/reviewed by an analyst.
    """
    for alert in ALERTS:
        if alert["id"] == alert_id:
            alert["resolved"] = True
            alert["resolvedBy"] = resolved_by
            alert["resolvedAt"] = datetime.datetime.now().isoformat()
            
            # Log resolution audit event
            AUDIT_LOGS.append({
                "id": len(AUDIT_LOGS) + 1,
                "eventType": "ALERT_RESOLVED",
                "actor": resolved_by,
                "entityType": "alert",
                "entityId": alert_id,
                "payload": {"status": "RESOLVED"},
                "createdAt": datetime.datetime.now().isoformat()
            })
            return alert
    return {"error": "Not Found", "message": f"Alert not found for ID: {alert_id}"}


# ─── TRANSACTION TOOLS ────────────────────────────────────────────────────────

@mcp.tool()
async def get_transaction_by_id(transaction_id: str) -> Any:
    """
    Fetch detailed information and scored status for a specific transaction by its UUID.
    """
    for txn in TRANSACTIONS:
        if txn["id"] == transaction_id:
            return txn
    return {"error": "Not Found", "message": f"Transaction not found for ID: {transaction_id}"}

@mcp.tool()
async def get_transactions(status: Optional[str] = None) -> Any:
    """
    Get a list of transactions, optionally filtered by status (PENDING, ALLOWED, BLOCKED, REVIEW).
    """
    if status:
        upper_status = status.upper()
        if upper_status not in ("PENDING", "ALLOWED", "BLOCKED", "REVIEW"):
            return {"error": "Invalid Status", "message": "Status must be one of: PENDING, ALLOWED, BLOCKED, REVIEW"}
        return [txn for txn in TRANSACTIONS if txn["status"] == upper_status]
    return TRANSACTIONS

@mcp.tool()
async def override_transaction_status(transaction_id: str, status: str, reason: Optional[str] = None) -> Any:
    """
    Manually override a transaction's status (e.g., set to ALLOWED or BLOCKED) and log an audit trail.
    """
    upper_status = status.upper()
    if upper_status not in ("ALLOWED", "BLOCKED", "REVIEW"):
        return {"error": "Invalid Status", "message": "Can only override to ALLOWED, BLOCKED, or REVIEW"}
    
    for txn in TRANSACTIONS:
        if txn["id"] == transaction_id:
            old_status = txn["status"]
            txn["status"] = upper_status
            txn["processedAt"] = datetime.datetime.now().isoformat()
            
            # Log override audit event
            AUDIT_LOGS.append({
                "id": len(AUDIT_LOGS) + 1,
                "eventType": "TRANSACTION_STATUS_OVERRIDDEN",
                "actor": "analyst-mcp",
                "entityType": "transaction",
                "entityId": transaction_id,
                "payload": {
                    "oldStatus": old_status,
                    "newStatus": upper_status,
                    "reason": reason
                },
                "createdAt": datetime.datetime.now().isoformat()
            })
            return txn
            
    return {"error": "Not Found", "message": f"Transaction not found for ID: {transaction_id}"}

@mcp.tool()
async def get_transaction_audit_logs(transaction_id: str) -> Any:
    """
    Fetch compliance and analyst override audit logs for a transaction.
    """
    return [log for log in AUDIT_LOGS if log["entityId"] == transaction_id and log["entityType"] == "transaction"]

@mcp.tool()
async def ingest_transaction(
    external_id: str,
    account_id: str,
    amount: float,
    merchant: Optional[str] = None,
    merchant_category: Optional[str] = None,
    location: Optional[str] = None,
    ip_address: Optional[str] = None,
    device_id: Optional[str] = None,
    channel: Optional[str] = None,
    hour_of_day: int = 12,
    day_of_week: int = 1,
    txn_per_hour: int = 1,
    distinct_merchants: int = 1,
    avg_txn_amount: float = 0.0,
    is_foreign: int = 0,
    is_new_device: int = 0,
    currency: str = "EUR"
) -> Any:
    """
    Ingest a new transaction payload. Returns immediately; fraud scoring processes asynchronously.
    """
    if channel and channel.lower() not in ("web", "mobile", "atm", "pos"):
        return {"error": "Invalid Channel", "message": "Channel must be one of: web, mobile, atm, pos"}
    
    # Generate new transaction ID
    txn_id = str(uuid.uuid4())
    
    # Simple Mock scoring heuristic
    score = 0.05
    status = "ALLOWED"
    
    # High amount or foreign/new device combos trigger anomaly
    if amount >= 1500.0:
        score = 0.85
        status = "BLOCK" if amount >= 5000.0 else "REVIEW"
    elif is_foreign == 1 and is_new_device == 1:
        score = 0.78
        status = "REVIEW"
    elif txn_per_hour > 10:
        score = 0.68
        status = "REVIEW"
        
    new_txn = {
        "id": txn_id,
        "externalId": external_id,
        "accountId": account_id,
        "amount": amount,
        "currency": currency,
        "merchant": merchant or "Generic Merchant",
        "merchantCategory": merchant_category or "MISC",
        "location": location or "Local Store",
        "ipAddress": ip_address or "127.0.0.1",
        "deviceId": device_id or "dev_unknown",
        "channel": channel.lower() if channel else "web",
        "status": status,
        "createdAt": datetime.datetime.now().isoformat(),
        "processedAt": datetime.datetime.now().isoformat()
    }
    TRANSACTIONS.append(new_txn)
    
    # Log ingestion audit event
    AUDIT_LOGS.append({
        "id": len(AUDIT_LOGS) + 1,
        "eventType": "TRANSACTION_INGESTED",
        "actor": "ingestion-service",
        "entityType": "transaction",
        "entityId": txn_id,
        "payload": {"amount": amount, "currency": currency},
        "createdAt": datetime.datetime.now().isoformat()
    })
    
    # If high risk, generate a new Alert
    if status in ("BLOCK", "REVIEW"):
        alert_id = str(uuid.uuid4())
        new_alert = {
            "id": alert_id,
            "transactionId": txn_id,
            "accountId": account_id,
            "modelVersion": "isolation_forest_v1.2.0",
            "fraudScore": score,
            "thresholdUsed": 0.6500,
            "featuresUsed": {
                "amount": amount,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "txn_per_hour": txn_per_hour,
                "distinct_merchants": distinct_merchants,
                "avg_txn_amount": avg_txn_amount,
                "is_foreign": is_foreign,
                "is_new_device": is_new_device
            },
            "decision": status,
            "resolved": False,
            "resolvedBy": None,
            "resolvedAt": None,
            "createdAt": datetime.datetime.now().isoformat()
        }
        ALERTS.append(new_alert)
        
        # Log flagging event
        AUDIT_LOGS.append({
            "id": len(AUDIT_LOGS) + 1,
            "eventType": "TRANSACTION_BLOCKED" if status == "BLOCK" else "TRANSACTION_FLAGGED_FOR_REVIEW",
            "actor": "fraud-detection-service",
            "entityType": "transaction",
            "entityId": txn_id,
            "payload": {
                "decision": status,
                "score": score,
                "alertId": alert_id
            },
            "createdAt": datetime.datetime.now().isoformat()
        })
        
    return {
        "id": txn_id,
        "status": status,
        "message": "Transaction received. Fraud scoring in progress."
    }


# ─── ANALYTICS TOOLS ──────────────────────────────────────────────────────────

@mcp.tool()
async def get_analytics_summary() -> Any:
    """
    Fetch aggregated daily transaction volumes, risk distribution, merchant categories, and location rankings.
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    return {
        "volumeOverTime": [
            {"date": yesterday, "total": 1420, "flagged": 24, "blocked": 8},
            {"date": today, "total": 1285, "flagged": 18, "blocked": 5}
        ],
        "categoryBreakdown": [
            {"category": "GROCERY", "count": 450, "fraudCount": 1},
            {"category": "ELECTRONICS", "count": 210, "fraudCount": 6},
            {"category": "FINANCIAL", "count": 85, "fraudCount": 11},
            {"category": "RESTAURANT", "count": 315, "fraudCount": 2},
            {"category": "TRAVEL", "count": 95, "fraudCount": 3}
        ],
        "riskDistribution": [
            {"range": "0.0 - 0.2", "count": 1100},
            {"range": "0.2 - 0.4", "count": 120},
            {"range": "0.4 - 0.6", "count": 45},
            {"range": "0.6 - 0.8", "count": 15},
            {"range": "0.8 - 1.0", "count": 5}
        ],
        "topFraudLocations": [
            {"location": "Unknown", "fraudCount": 12, "totalCount": 24},
            {"location": "London, UK", "fraudCount": 4, "totalCount": 850},
            {"location": "New York, USA", "fraudCount": 3, "totalCount": 120},
            {"location": "Berlin, DE", "fraudCount": 2, "totalCount": 95},
            {"location": "Tokyo, JP", "fraudCount": 2, "totalCount": 80}
        ]
    }


# ─── ML MONITORING TOOLS ──────────────────────────────────────────────────────

@mcp.tool()
async def get_ml_health() -> Any:
    """
    Check the FastAPI ML Service health and check if the Isolation Forest model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": "isolation_forest_v1.2.0"
    }

@mcp.tool()
async def get_ml_model_info() -> Any:
    """
    Retrieve loaded model metadata (algorithm, version, feature order, metrics).
    """
    return {
        "version": "isolation_forest_v1.2.0",
        "algorithm": "Isolation Forest (scikit-learn)",
        "feature_names": [
            "amount",
            "hour_of_day",
            "day_of_week",
            "txn_per_hour",
            "distinct_merchants",
            "avg_txn_amount",
            "is_foreign",
            "is_new_device"
        ],
        "trained_at": (datetime.datetime.now() - datetime.timedelta(days=15)).isoformat(),
        "metrics": {
            "accuracy": 0.9845,
            "f1_score": 0.8920,
            "precision": 0.9110,
            "recall": 0.8740,
            "training_samples": 450000
        },
        "is_loaded": True
    }

@mcp.tool()
async def get_ml_drift_report() -> Any:
    """
    Check feature drift alerts and rolling statistics comparing production features against baseline.
    """
    return {
        "alerts": [
            {
                "feature": "txn_per_hour",
                "z_score": 3.12,
                "p_value": 0.0018,
                "status": "DRIFT_DETECTED"
            }
        ],
        "stats": {
            "amount": {"mean": 112.50, "std": 320.00, "baseline_mean": 108.20},
            "txn_per_hour": {"mean": 4.80, "std": 2.10, "baseline_mean": 2.30},
            "is_foreign": {"mean": 0.12, "std": 0.32, "baseline_mean": 0.10}
        }
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)
