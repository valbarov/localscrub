"""Clinical note templates with entity slots.

A template body is plain text with ``{slot}`` placeholders. Slots come in two
kinds:

- **entity slots** (in ``ENTITY_SLOTS``) fill with a fabricated identifier and
  contribute a gold span;
- **filler slots** (in ``FILLER_BANKS``) fill with non-PHI clinical phrasing
  and contribute nothing to the ground truth — including deliberate hard
  negatives (ages under 90, bare years, blood pressures) that a detector must
  *not* flag.

A trailing digit distinguishes repeated independent slots (``{provider2}``);
the digit is stripped before lookup. Repeating the *same* slot name reuses the
same value within a note, so coreference ("the patient's name appears three
times") is represented honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

from localscrub.entities import EntityType

#: slot key (digit-suffix stripped) -> gold entity type
ENTITY_SLOTS: dict[str, EntityType] = {
    "patient": EntityType.NAME,
    "provider": EntityType.NAME,
    "relative": EntityType.NAME,
    "address": EntityType.GEO_SUBDIVISION,
    "dob": EntityType.DATE,
    "visit_date": EntityType.DATE,
    "admit_date": EntityType.DATE,
    "discharge_date": EntityType.DATE,
    "followup_date": EntityType.DATE,
    "age89": EntityType.DATE,
    "phone": EntityType.PHONE,
    "fax": EntityType.FAX,
    "email": EntityType.EMAIL,
    "ssn": EntityType.SSN,
    "mrn": EntityType.MRN,
    "plan_id": EntityType.HEALTH_PLAN_ID,
    "account": EntityType.ACCOUNT_NUMBER,
    "url": EntityType.URL,
    "ip": EntityType.IP_ADDRESS,
    "credit_card": EntityType.CREDIT_CARD,
}

#: filler slot key -> phrase bank (non-PHI, includes hard negatives)
FILLER_BANKS: dict[str, tuple[str, ...]] = {
    "age_u90": ("42-year-old", "57-year-old", "68-year-old", "74-year-old", "81-year-old"),
    "complaint": (
        "worsening shortness of breath",
        "intermittent chest pain",
        "poorly controlled blood glucose",
        "a productive cough of three days' duration",
        "progressive lower-extremity edema",
        "recurrent dizziness on standing",
    ),
    "diagnosis": (
        "community-acquired pneumonia",
        "type 2 diabetes mellitus",
        "congestive heart failure, NYHA class II",
        "chronic obstructive pulmonary disease",
        "atrial fibrillation with controlled ventricular response",
        "stage 3 chronic kidney disease",
    ),
    "medication": (
        "metformin 500 mg twice daily",
        "lisinopril 10 mg daily",
        "furosemide 40 mg each morning",
        "apixaban 5 mg twice daily",
        "atorvastatin 20 mg nightly",
    ),
    "bp": ("118/72", "126/78", "134/82", "142/88"),
    "hr": ("64", "72", "78", "88"),
    "specialty": ("cardiology", "endocrinology", "pulmonology", "nephrology"),
    "unit": ("4 West", "the medical ICU", "the observation unit", "6 North"),
    "year": ("2018", "2019", "2020", "2021"),
    "duration": ("two weeks", "one month", "three months", "six weeks"),
}


@dataclass(frozen=True, slots=True)
class NoteTemplate:
    """A named note skeleton; ``note_type`` groups template variants."""

    note_type: str
    body: str


TEMPLATES: tuple[NoteTemplate, ...] = (
    NoteTemplate(
        note_type="progress_note",
        body="""PROGRESS NOTE

Patient: {patient}
MRN: {mrn}
DOB: {dob}
Seen on: {visit_date}

{patient} is a {age_u90} patient presenting with {complaint}. History of
{diagnosis} since {year}; continues {medication}. BP {bp}, HR {hr}.
Discussed results by phone at {phone}. Plan: continue current regimen and
return on {followup_date}. Questions may be sent to {email}.

{provider}, MD
""",
    ),
    NoteTemplate(
        note_type="progress_note",
        body="""CLINIC NOTE — {specialty}

Name: {patient}    Record #: {mrn}
Visit date: {visit_date}

The patient, {age89}, was accompanied by {relative} and reports {complaint}
of {duration}. Longstanding {diagnosis}; {medication} unchanged. BP {bp}.
Home address confirmed as {address}. Follow-up {followup_date};
front desk will call {phone} to confirm.

Dictated by {provider}, MD
""",
    ),
    NoteTemplate(
        note_type="discharge_summary",
        body="""DISCHARGE SUMMARY

Patient: {patient} (MRN: {mrn})
DOB: {dob}
Admitted: {admit_date}    Discharged: {discharge_date}

{patient} was admitted to {unit} with {complaint}, treated for {diagnosis},
and discharged in improved condition on {medication}. Discharge vitals:
BP {bp}, HR {hr}. Member ID: {plan_id}. Follow-up with {provider} in
{specialty} on {followup_date}; the office ({phone}) will arrange transport
to {address}. After-visit summary posted to {url}.
""",
    ),
    NoteTemplate(
        note_type="referral_letter",
        body="""Dear Dr. {provider2},

Thank you for seeing {patient} (DOB {dob}, MRN: {mrn}) in {specialty}
consultation for {complaint}. The patient is {age_u90}, with a history of
{diagnosis} managed with {medication} since {year}.

Our direct line is {phone}, and records may be faxed to {fax}. The
patient prefers contact at {email}.

Sincerely,
{provider}, MD
{address}
""",
    ),
    NoteTemplate(
        note_type="nursing_note",
        body="""NURSING NOTE — {unit}

Pt {patient}, MRN: {mrn}, {age89}. Resting comfortably; BP {bp}, HR {hr}.
Daughter {relative} at bedside, reachable at {phone}. Pt reports {complaint}
resolved. Next of kin contact updated; emergency contact remains {relative}
({email}). Social Security card on file (SSN {ssn}) returned to patient.
Seen by {provider} on rounds {visit_date}.
""",
    ),
    NoteTemplate(
        note_type="billing_note",
        body="""BILLING / REGISTRATION RECORD

Guarantor: {patient}
Account #: {account}
Member ID: {plan_id}
Statement address: {address}

Copay for the {visit_date} visit charged to card {credit_card}. Patient
called from {phone} to confirm the balance; SSN {ssn} verified for identity.
Statement portal: {url}. Receipt emailed to {email}.
""",
    ),
    NoteTemplate(
        note_type="telehealth_note",
        body="""TELEHEALTH ENCOUNTER

Patient: {patient} (MRN: {mrn}, DOB {dob})
Session date: {visit_date}

Video visit completed from patient IP {ip} via {url}. Connection stable.
{patient} reports {complaint}; reviewed {medication} adherence. BP (home
cuff) {bp}. Verified callback number {phone} and email {email}. Follow-up
scheduled {followup_date} with {provider}.
""",
    ),
)

#: distinct note types, in first-appearance order
NOTE_TYPES: tuple[str, ...] = tuple(dict.fromkeys(t.note_type for t in TEMPLATES))
