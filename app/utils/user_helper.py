def calculate_remaining_seconds(user_data: dict) -> float:
    """
    Calculate remaining credits (now simplified credit-based system)
    All users (free and premium) use credits - no more separate free seconds
    """
    credits = user_data.get("credits", 0)
    # Simply return credits as is (credit-based system)
    return float(credits)