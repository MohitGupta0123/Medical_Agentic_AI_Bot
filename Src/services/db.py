# services/db.py
import os
from contextlib import contextmanager
from typing import Iterator, Optional
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, text, Integer, String, Text, ForeignKey
)
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, Mapped, mapped_column

# ────────────────────────────────────────────────────────────────────────────────
# Env: use your Supabase pooled URL (PgBouncer port 6543) with sslmode=require
# e.g. postgresql+psycopg://USER:PASSWORD@aws-0-...pooler.supabase.com:6543/postgres?sslmode=require
# Set it in HF Spaces → Settings → Secrets as DATABASE_URL
# ────────────────────────────────────────────────────────────────────────────────

# load_dotenv(dotenv_path='Secrets/.env')

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is not set")

# Ensure SQLAlchemy uses psycopg3 driver
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Ensure SSL
if "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = f"{DATABASE_URL}{sep}sslmode=require"

# ────────────────────────────────────────────────────────────────────────────────
# SQLAlchemy setup
# ────────────────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # drops dead conns
    pool_size=5,          # conservative for HF Spaces
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

# ────────────────────────────────────────────────────────────────────────────────
# ORM Models
# ────────────────────────────────────────────────────────────────────────────────
class Doctor(Base):
    __tablename__ = "doctors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    specialization: Mapped[str] = mapped_column(String(255), nullable=False)
    available: Mapped[int] = mapped_column(Integer, default=1)
    # store timestamps as ISO strings for compatibility with your existing code
    last_booked_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    registered_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    doctor_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("doctors.id"))

class Medicine(Base):
    __tablename__ = "medicines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    stock: Mapped[int] = mapped_column(Integer, nullable=False)

# ────────────────────────────────────────────────────────────────────────────────
# Public API (similar names)
# ────────────────────────────────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a SQLAlchemy session (use this in services)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Compatibility shim: some of your modules import get_connection()
# This returns a Session, not a sqlite3 connection. Update callers to use get_session().
def get_connection() -> Session:
    return SessionLocal()

def seed_data():
    """Populate doctors and medicines if empty."""
    from sqlalchemy import select

    doctors = [
        ("Dr. Ananya Sharma", "Cardiology"),
        ("Dr. Monika Verma", "Neurology"),
        ("Dr. Kiara Joshi", "Orthopedics"),
        ("Dr. Vaani Gupta", "General Medicine"),
        ("Dr. Aarti Agarwal", "Otolaryngology (ENT)"),
        ("Dr. Nidhi Mishra", "Dermatology"),
        ("Dr. Riya Srivastava", "Pediatrics"),
        ("Dr. Sneha Bhardwaj", "Medical Oncology"),
        ("Dr. Isha Agarwal", "Endocrinology"),
        ("Dr. Ishita Saxena", "Psychiatry"),
        ("Dr. Swati Tripathi", "Gastroenterology"),
        ("Dr. Kriti Bansal", "Nephrology"),
        ("Dr. Shalini Mehra", "Pulmonology"),
        ("Dr. Kritika Chaturvedi", "Rheumatology"),
        ("Dr. Aparna Sharma", "Obstetrics & Gynecology"),
        ("Dr. Nandita Pandey", "Hematology"),
        ("Dr. Preeti Jain", "Ophthalmology"),
        ("Dr. Disha Goyal", "Radiology"),
        ("Dr. Shruti Shukla", "Urology"),
        ("Dr. Bhavya Dubey", "Infectious Diseases"),
        ("Dr. Priyanka Tiwari", "Critical Care"),
        ("Dr. Vandana Srivastava", "Anesthesiology"),
        ("Dr. Shweta Tripathi", "Geriatrics"),
        ("Dr. Monika Khandelwal", "Pathology"),
        ("Dr. Ritika Gupta", "Allergy & Immunology"),
        ("Dr. Kusum Agarwal", "General Surgery"),   # single surgery doc for plastic/vascular/etc.
        ("Dr. Poonam Verma", "Palliative Care"),
        ("Dr. Sakshi Sharma", "Sports Medicine"),
        ("Dr. Rachna Joshi", "Reproductive Endocrinology & Infertility"),
        ("Dr. Garima Bhardwaj", "Nuclear Medicine"),
        ("Dr. Aishwarya Srivastava", "Physical Medicine & Rehabilitation"),
        ("Dr. Jyoti Jain", "Dentistry"),
        ("Dr. Manisha Gupta", "Public Health & Preventive Medicine"),
        ("Dr. Kavita Tyagi", "Occupational Medicine"),
        ("Dr. Rupal Mishra", "Clinical Genetics & Genomic Medicine"),
        ("Dr. Neelam Rawat", "Nutrition & Dietetics"),
        ("Dr. Simran Shukla", "Audiology & Speech Therapy"),
        ("Dr. Vaishali Tripathi", "Transfusion Medicine (Blood Bank)"),
        ("Dr. Anupriya Saxena", "Hepatology"),
        ("Dr. Radhika Chaturvedi", "Clinical Pharmacology"),
        ("Dr. Divya Agarwal", "Radiation Oncology"),
        ("Dr. Meera Srivastava", "Interventional Radiology"),
        ("Dr. Kavya Jain", "Clinical Microbiology"),
        ("Dr. Aarti Goyal", "Clinical Biochemistry"),
        ("Dr. Tanisha Sharma", "Neonatology"),
        ("Dr. Prerna Tiwari", "Sleep Medicine"),
        ("Dr. Niharika Gupta", "Tropical Medicine"),
        ("Dr. Deeksha Verma", "Travel Medicine"),
        ("Dr. Ishani Srivastava", "Maternal–Fetal Medicine"),
        ("Dr. Prachi Bhardwaj", "Neuroradiology"),
    ]

    medicines = [
        ("Aspirin", 100), ("Paracetamol", 200), ("Metformin", 150),
        ("Ibuprofen", 120), ("Amoxicillin", 80), ("Ciprofloxacin", 90),
        ("Azithromycin", 75), ("Lisinopril", 60), ("Atorvastatin", 90),
        ("Omeprazole", 75), ("Losartan", 50), ("Metoprolol", 40),
        ("Furosemide", 60), ("Levothyroxine", 100), ("Insulin (Human)", 80),
        ("Salbutamol", 90), ("Budesonide", 70), ("Ondansetron", 65),
        ("Clopidogrel", 85), ("Warfarin", 40), ("Heparin", 50),
        ("Prednisolone", 55), ("Hydroxychloroquine", 60), ("Artemether-Lumefantrine", 45),
        ("Ranitidine", 70), ("Calcium Carbonate + Vitamin D3", 90), ("Iron + Folic Acid", 100),
        ("Multivitamin Tablet", 120), ("Oral Rehydration Salts (ORS)", 150), ("Zinc Sulfate", 80),
        ("Amlodipine", 85), ("Glimepiride", 65), ("Doxycycline", 55),
        ("Fluconazole", 50), ("Aciclovir", 40), ("Oseltamivir", 35),
        ("Cetirizine", 100), ("Chlorpheniramine", 80), ("Hydralazine", 45),
        ("Nitroglycerin", 30), ("Digoxin", 25), ("Phenytoin", 40),
        ("Sodium Valproate", 35), ("Carbamazepine", 30), ("Diazepam", 50),
        ("Lorazepam", 40), ("Haloperidol", 25), ("Olanzapine", 30),
    ]

    with get_session() as s:
        # Doctors
        count = s.execute(text("SELECT COUNT(*) FROM doctors")).scalar_one()
        if count == 0:
            s.bulk_save_objects([Doctor(name=n, specialization=sp) for n, sp in doctors])

        # Medicines
        count = s.execute(text("SELECT COUNT(*) FROM medicines")).scalar_one()
        if count == 0:
            s.bulk_save_objects([Medicine(name=n, stock=stk) for n, stk in medicines])

# Utility (kept for compatibility with your old helpers that used sqlite row)
def row_to_dict(row) -> dict | None:
    if row is None:
        return None
    # For ORM objects, expose __dict__-like mapping
    if hasattr(row, "__table__"):
        # only include column attributes
        return {c.name: getattr(row, c.name) for c in row.__table__.columns}
    if isinstance(row, dict):
        return row
    return dict(row)
