import json
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


def rule_based_recommendation(
    predicted_demand,
    current_stock,
    inventory_status,
    expiry_status,
    promotion,
    holiday,
    waste_risk,
):
    recommendation = []

    if inventory_status == "Stock Shortage":
        recommendation.append(
            "Forecasted demand is higher than the available inventory."
        )

        if holiday:
            recommendation.append(
                "Demand may increase further during the holiday period."
            )

        recommendation.append(
            "Increase replenishment to avoid stock shortages."
        )

    elif inventory_status == "Overstock":
        recommendation.append(
            "Current inventory exceeds the predicted demand."
        )

        if waste_risk == "High":
            recommendation.append(
                "Products are at high risk of becoming waste."
            )
            recommendation.append(
                "Launch an immediate discount campaign and redistribute "
                "stock to nearby stores."
            )

        elif waste_risk == "Medium":
            recommendation.append(
                "Monitor inventory closely and consider promotional activities."
            )

        else:
            recommendation.append(
                "Reduce future order quantities."
            )

    else:
        recommendation.append(
            "Inventory levels are aligned with predicted demand."
        )

        if promotion:
            recommendation.append(
                "Continue monitoring the current promotion before placing "
                "the next order."
            )

        else:
            recommendation.append(
                "Maintain the current replenishment strategy."
            )

    return " ".join(recommendation)


def generate_llm_recommendation(
    predicted_demand,
    current_stock,
    inventory_status,
    expiry_status,
    promotion,
    holiday,
    waste_risk,
    base_recommendation,
):
    prompt = f"""
You are an intelligent supermarket inventory decision-support assistant.

Use the structured forecasting and inventory information below to provide
a short, practical, context-aware recommendation for a retail manager.

Predicted demand: {predicted_demand:.2f}
Current stock: {current_stock}
Inventory status: {inventory_status}
Expiry status: {expiry_status}
Promotion active: {bool(promotion)}
Holiday period: {bool(holiday)}
Waste risk: {waste_risk}

Rule-based recommendation:
{base_recommendation}

Instructions:
- Give a clear business recommendation.
- Explain briefly why the action is appropriate.
- Mention inventory-related waste risk when relevant.
- Do not claim that actual food waste has been measured.
- Do not invent new numerical values.
- Keep the answer concise, around 60 to 100 words.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=request_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

            llm_text = result.get(
                "response",
                ""
            ).strip()

            if not llm_text:
                raise ValueError(
                    "Empty response received from Ollama."
                )

            return {
                "recommendation": llm_text,
                "recommendation_source": "local_llm_qwen2.5"
            }

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ):

        return {
            "recommendation": base_recommendation,
            "recommendation_source": "rule_based_fallback"
        }


def generate_recommendation(
    predicted_demand,
    current_stock,
    inventory_status,
    expiry_status,
    promotion,
    holiday,
    waste_risk,
):
    base_recommendation = rule_based_recommendation(
        predicted_demand=predicted_demand,
        current_stock=current_stock,
        inventory_status=inventory_status,
        expiry_status=expiry_status,
        promotion=promotion,
        holiday=holiday,
        waste_risk=waste_risk,
    )

    return generate_llm_recommendation(
        predicted_demand=predicted_demand,
        current_stock=current_stock,
        inventory_status=inventory_status,
        expiry_status=expiry_status,
        promotion=promotion,
        holiday=holiday,
        waste_risk=waste_risk,
        base_recommendation=base_recommendation,
    )