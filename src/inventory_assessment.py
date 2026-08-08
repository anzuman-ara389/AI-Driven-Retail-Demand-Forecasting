def assess_inventory(predicted_demand, current_stock, expiry_days):

    inventory_gap = current_stock - predicted_demand

    # Inventory status
    if predicted_demand > current_stock:
        inventory_status = "Stock Shortage"

    elif inventory_gap >= current_stock * 0.30:
        inventory_status = "Overstock"

    else:
        inventory_status = "Balanced Inventory"

    # Expiry status
    if expiry_days <= 2:
        expiry_status = "Critical"

    elif expiry_days <= 5:
        expiry_status = "Short Expiry"

    else:
        expiry_status = "Normal"

    # Waste risk
    if inventory_status == "Overstock" and expiry_status == "Critical":
        waste_risk = "High"

    elif inventory_status == "Overstock" or expiry_status == "Short Expiry":
        waste_risk = "Medium"

    else:
        waste_risk = "Low"

    # Recommendation
    if inventory_status == "Stock Shortage":

        suggested_action = "Increase Order Quantity"

    elif waste_risk == "High":

        suggested_action = "Launch Immediate Promotion"

    elif inventory_status == "Overstock":

        suggested_action = "Redistribute Inventory"

    else:

        suggested_action = "Maintain Current Inventory"

    return {
        "inventory_status": inventory_status,
        "expiry_status": expiry_status,
        "waste_risk": waste_risk,
        "inventory_gap": round(inventory_gap, 2),
        "suggested_action": suggested_action,
    }