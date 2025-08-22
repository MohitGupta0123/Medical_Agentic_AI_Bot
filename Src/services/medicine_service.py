# Src/services/medicine_service.py
from typing import Optional
from sqlalchemy import select, func
from .db import get_session, Medicine


# --------------------------
# Check Medicine Availability
# --------------------------
def check_medicine_availability(name: str) -> str:
    """
    Check if a medicine is available in stock.
    - Supports partial and case-insensitive search.
    - Returns human-readable message.
    """
    with get_session() as s:
        # case-insensitive partial search
        q = (
            select(Medicine)
            .where(func.lower(Medicine.name).like(f"%{name.lower()}%"))
            .limit(1)
        )
        med: Optional[Medicine] = s.execute(q).scalar_one_or_none()

        if med:
            if med.stock > 0:
                return f"Medicine '{med.name}' is available with {med.stock} units."
            else:
                return f"Medicine '{med.name}' is currently out of stock."
        else:
            return f"No medicine found matching '{name}'. Please check spelling or try another name."


# --------------------------
# Deduct Medicine Stock
# --------------------------
def deduct_medicine_stock(name: str, quantity: int = 1) -> bool:
    """
    Deduct specified quantity of medicine stock after issuing to patient.
    - If stock is insufficient or medicine not found, returns False.
    - Otherwise, deducts and returns True.
    """
    if quantity <= 0:
        return True  # no-op

    with get_session() as s:
        # case-insensitive exact match
        q = select(Medicine).where(func.lower(Medicine.name) == name.lower()).limit(1)
        med: Optional[Medicine] = s.execute(q).scalar_one_or_none()

        if not med:
            return False  # Medicine not found

        if med.stock < quantity:
            return False  # Insufficient stock

        med.stock -= quantity
        s.add(med)
        # session commits on context exit
        return True


# --------------------------
# Refill Medicine Stock
# --------------------------
def refill_medicine_stock(name: str, quantity: int) -> bool:
    """
    Refill or add new medicine stock.
    - If medicine exists → increase stock.
    - If not → insert new medicine record.
    """
    if quantity <= 0:
        return True  # no-op

    with get_session() as s:
        q = select(Medicine).where(func.lower(Medicine.name) == name.lower()).limit(1)
        med: Optional[Medicine] = s.execute(q).scalar_one_or_none()

        if med:
            med.stock += quantity
            s.add(med)
        else:
            s.add(Medicine(name=name, stock=quantity))

        # commit on context exit
        return True
