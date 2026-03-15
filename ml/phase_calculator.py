from datetime import date


def get_cycle_phase(last_period_date: str, cycle_length: int = 28) -> str:

    last_period = date.fromisoformat(last_period_date)

    if last_period > date.today():
        raise ValueError("Last period date cannot be in the future")
    if cycle_length < 21 or cycle_length > 35:
        raise ValueError("Cycle length must be between 21–35 days")

    day_of_cycle = (date.today() - last_period).days % cycle_length

    if day_of_cycle < 5:
        return "menstrual"
    elif day_of_cycle < 14:
        return "follicular"
    elif day_of_cycle < 17:
        return "ovulatory"
    else:
        return "luteal"


def get_cycle_phase_from_day(day_of_cycle: int, cycle_length: int = 28) -> str:
    """
    For dataset use — takes day number directly instead of date.
    Used in bias_analysis to assign phases row by row.
    """
    if not (0 <= day_of_cycle < cycle_length):
        raise ValueError(f"day_of_cycle must be between 0 and {cycle_length - 1}")

    if day_of_cycle < 5:
        return "menstrual"
    elif day_of_cycle < 14:
        return "follicular"
    elif day_of_cycle < 17:
        return "ovulatory"
    else:
        return "luteal"


# Quick test
if __name__ == "__main__":
    phase = get_cycle_phase("2026-03-01", 28)
    print(f"Current phase: {phase}")

    # Edge case tests
    try:
        get_cycle_phase("2026-04-01", 28)   # future date
    except ValueError as e:
        print(f"Caught: {e}")

    try:
        get_cycle_phase("2026-03-01", 18)   # abnormal cycle
    except ValueError as e:
        print(f"Caught: {e}")