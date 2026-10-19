import json
import re
import subprocess
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from smolagents import Tool

from event_logging import log_tool_calls


# Case IDs encode the antrag type in their directory-name prefix, e.g.
# "dienstreiseantrag-03", "auslagenerstattung-03", or "beschaffungsantrag-03".
# This maps each known prefix to a short internal case-type key used to pick
# the matching policy set, system prompt, and search terms.
CASE_TYPE_PREFIXES = {
    "dienstreiseantrag": "business_trip",
    "auslagenerstattung": "expense_reimbursement",
    "beschaffungsantrag": "procurement",
}


def case_type_for(case_id: str) -> str:
    """Return the case type ("business_trip", "expense_reimbursement", or
    "procurement") encoded in a case directory name's prefix.

    Raises ValueError for a case_id whose prefix does not match any known
    antrag type, so an unrecognised input directory fails loudly instead of
    silently falling back to the wrong policy set or prompt.
    """
    for prefix, case_type in CASE_TYPE_PREFIXES.items():
        if case_id == prefix or case_id.startswith(f"{prefix}-"):
            return case_type
    raise ValueError(
        f"Unknown case type for {case_id!r}; expected a directory name starting with one of "
        f"{sorted(CASE_TYPE_PREFIXES)}."
    )


BUSINESS_TRIP_POLICIES = {
    "advance_approval": (
        "Dienstreiseanträge sind vor Reiseantritt einzureichen und zu genehmigen. "
        "Eine nachträgliche Genehmigung erfordert ein gesondertes Ausnahmeverfahren."
    ),
    "foreign_travel": (
        "Bei Auslandsreisen müssen das Auslandsfeld korrekt ausgefüllt, die zuständige "
        "Personalabteilung beteiligt und eine A1-Bescheinigung mit ausreichendem Vorlauf beantragt werden."
    ),
    "travel_evidence": (
        "Hin- und Rückreise, Reisedaten, Zweck und voraussichtliche Gesamtkosten müssen "
        "vollständig und widerspruchsfrei dokumentiert sein."
    ),
    "hotel_costs": (
        "Übernachtungskosten müssen innerhalb der einschlägigen Höchstgrenze liegen oder "
        "mit einem nachvollziehbaren Ausnahmegrund belegt werden."
    ),
    "private_extension": (
        "Bei privaten Reiseanteilen dürfen der Universität keine Mehrkosten entstehen. "
        "Private Übernachtungen und Mehrpreise sind getrennt auszuweisen und selbst zu tragen."
    ),
    "price_comparison": (
        "Bei relevanten Alternativen ist ein nachvollziehbarer Preisvergleich erforderlich, "
        "insbesondere bei Flug gegenüber Bahn oder einer privat verlängerten Rückreise."
    ),
    "invoice_recipient": (
        "Rechnungen sollen an die Universität adressiert sein. Bei privaten Rechnungsadressaten "
        "kann ein ordnungsgemäßer Ersatzbeleg erforderlich sein."
    ),
    "double_funding": (
        "Dieselben Kostenpositionen dürfen nicht zugleich durch ein Stipendium, einen Dritten "
        "und die Universität finanziert oder erstattet werden."
    ),
    "optional_events": (
        "Nicht dienstlich erforderliche optionale Programmpunkte, etwa Abendveranstaltungen, "
        "dürfen nicht ohne gesonderte Begründung als erstattungsfähige Kosten beantragt werden."
    ),
}

EXPENSE_REIMBURSEMENT_POLICIES = {
    "proof_of_purchase": (
        "Für eine Erstattung ist ein ordnungsgemäßer Zahlungsbeleg erforderlich. Ein reiner "
        "Kassenbon ohne Artikelbezeichnung ist zulässig, wenn der Verwendungszweck handschriftlich "
        "oder anderweitig nachvollziehbar auf dem Beleg vermerkt ist."
    ),
    "invoice_recipient": (
        "Rechnungen sollen an die Universität adressiert oder zumindest der dienstliche Bezug "
        "erkennbar sein. Eine Rechnung an eine Privatperson ohne jeglichen dienstlichen Bezug "
        "auf dem Beleg ist nicht erstattungsfähig."
    ),
    "proforma_documents": (
        "Pro-forma-Rechnungen, Angebote oder Warenkörbe gelten nicht als Zahlungsbeleg. "
        "Erforderlich ist die tatsächliche, als bezahlt gekennzeichnete Rechnung."
    ),
    "duplicate_claims": (
        "Dieselbe Bestellung darf nicht mehrfach unter verschiedenen Belegen oder Rechnungsnummern "
        "zur Erstattung eingereicht werden."
    ),
    "credit_notes": (
        "Werden Teile einer Bestellung storniert oder gutgeschrieben, ist nur der Nettobetrag nach "
        "Abzug der Gutschrift erstattungsfähig."
    ),
    "private_use_items": (
        "Gegenstände oder Verbrauchsmaterialien ohne erkennbaren dienstlichen Bezug (z. B. für den "
        "privaten Gebrauch oder mit einer Personenbezeichnung statt einer Artikelbeschreibung) sind "
        "nicht erstattungsfähig, unabhängig vom Rechnungsempfänger."
    ),
    "foreign_currency": (
        "Im Ausland oder in Fremdwährung entstandene Kosten sind erstattungsfähig, wenn Beleg und "
        "Zahlungsnachweis (z. B. Kontoauszug) gemeinsam vorgelegt werden."
    ),
}

PROCUREMENT_POLICIES = {
    "direct_award_threshold": (
        "Bestellungen bis 1.000 EUR netto können im Wege der Direktvergabe ohne Vergleichsangebote "
        "erfolgen."
    ),
    "comparison_offers_required": (
        "Ab 1.000 EUR netto sind grundsätzlich mindestens drei Vergleichsangebote einzuholen und im "
        "Vergabevermerk zu dokumentieren; fehlt ein Angebot, ist eine nachvollziehbare Begründung "
        "erforderlich."
    ),
    "framework_contracts": (
        "Artikel aus einem bestehenden Rahmenvertrag (z. B. über das Warenwirtschafts-/Bestellsystem "
        "WPS) sind ohne gesonderten Preisvergleich zu beschaffen; ein zusätzlicher Preisvergleich "
        "außerhalb des Rahmenvertrags ist bei Rahmenvertragsartikeln nicht erforderlich."
    ),
    "cheapest_offer": (
        "Den Zuschlag erhält grundsätzlich das wirtschaftlichste, in der Regel das günstigste, "
        "Angebot bei gleicher technischer Eignung; Abweichungen sind im Vergabevermerk zu begründen."
    ),
    "consistency_check": (
        "Die im Antrag oder Vergabevermerk genannten Beträge müssen mit der Summe der Einzelpositionen "
        "und den vorgelegten Angebots- und Rechnungsdokumenten übereinstimmen."
    ),
    "invoice_matching": (
        "Rechnungen müssen dem beauftragten Angebot entsprechen; mehrere Teilrechnungen müssen in "
        "Summe dem gewählten Angebot entsprechen."
    ),
}

POLICIES_BY_CASE_TYPE = {
    "business_trip": BUSINESS_TRIP_POLICIES,
    "expense_reimbursement": EXPENSE_REIMBURSEMENT_POLICIES,
    "procurement": PROCUREMENT_POLICIES,
}


def _safe_case(root: Path, allowed_case: str, case_id: str) -> Path:
    if case_id != allowed_case:
        raise ValueError(f"This tool is restricted to case {allowed_case!r}.")
    case = (root / case_id).resolve()
    if case.parent != root.resolve() or not case.is_dir():
        raise ValueError(f"Unknown case: {case_id}")
    return case


def _safe_pdf(root: Path, allowed_case: str, case_id: str, filename: str) -> Path:
    case = _safe_case(root, allowed_case, case_id)
    if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
        raise ValueError("filename must name one PDF directly inside the case directory.")
    pdf = (case / filename).resolve()
    if pdf.parent != case or not pdf.is_file():
        raise ValueError(f"Unknown PDF in {case_id}: {filename}")
    return pdf


@lru_cache(maxsize=128)
def _pdf_text(pdf: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = result.stdout.strip()
    if not text:
        raise ValueError(f"No text could be extracted from {pdf.name}.")
    return text


class CaseTool(Tool):
    def __init__(self, root: Path, allowed_case: str):
        super().__init__()
        self.root = root.resolve()
        self.allowed_case = allowed_case


class ListCaseDocumentsTool(CaseTool):
    name = "list_case_documents"
    description = (
        "List the PDFs available for the current application (business trip, expense "
        "reimbursement, or procurement)."
    )
    inputs = {
        "case_id": {
            "type": "string",
            "description": (
                "Application ID, for example dienstreiseantrag-03, auslagenerstattung-03, "
                "or beschaffungsantrag-03."
            ),
        }
    }
    output_type = "string"

    def forward(self, case_id: str) -> str:
        case = _safe_case(self.root, self.allowed_case, case_id)
        documents = [
            {"filename": path.name, "size_bytes": path.stat().st_size}
            for path in sorted(case.glob("*.pdf"))
        ]
        return json.dumps(documents, ensure_ascii=False)


class ReadPdfTool(CaseTool):
    name = "read_pdf"
    description = (
        "Extract layout-preserving text from one PDF. Use list_case_documents first, "
        "then read documents needed to verify the application."
    )
    inputs = {
        "case_id": {"type": "string", "description": "Current application ID."},
        "filename": {"type": "string", "description": "PDF filename returned by list_case_documents."},
    }
    output_type = "string"

    def __init__(self, root: Path, allowed_case: str):
        super().__init__(root, allowed_case)
        self.read_documents: set[str] = set()

    def forward(self, case_id: str, filename: str) -> str:
        text = _pdf_text(_safe_pdf(self.root, self.allowed_case, case_id, filename))
        self.read_documents.add(filename)
        return text


class SearchCaseTool(CaseTool):
    name = "search_case"
    description = (
        "Search all PDFs in the current case for words or phrases. Returns ranked, cited "
        "snippets. Use German and document-specific terms, such as Rückreise, Finanzierung, or A1 "
        "for business trips; Rechnung, Gutschrift, or Kontoauszug for expense reimbursements; or "
        "Vergabevermerk, Angebot, or Rahmenvertrag for procurement."
    )
    inputs = {
        "case_id": {"type": "string", "description": "Current application ID."},
        "query": {"type": "string", "description": "Space-separated search words or a phrase."},
        "max_results": {"type": "integer", "description": "Maximum number of snippets, between 1 and 20."},
    }
    output_type = "string"

    def forward(self, case_id: str, query: str, max_results: int) -> str:
        case = _safe_case(self.root, self.allowed_case, case_id)
        terms = [term.casefold() for term in re.findall(r"\w+", query) if len(term) > 1]
        if not terms:
            raise ValueError("query must contain at least one searchable word.")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20.")

        matches = []
        for pdf in sorted(case.glob("*.pdf")):
            lines = _pdf_text(pdf).splitlines()
            for index, line in enumerate(lines):
                folded = line.casefold()
                score = sum(folded.count(term) for term in terms)
                if score:
                    start = max(0, index - 1)
                    end = min(len(lines), index + 2)
                    snippet = " ".join(part.strip() for part in lines[start:end] if part.strip())
                    matches.append(
                        {
                            "score": score,
                            "document": pdf.name,
                            "line": index + 1,
                            "snippet": snippet,
                        }
                    )
        matches.sort(key=lambda item: (-item["score"], item["document"], item["line"]))
        return json.dumps(matches[:max_results], ensure_ascii=False)


class LookupPolicyTool(Tool):
    name = "lookup_policy"
    description = (
        "Look up applicable rules for the current application type (business trip, expense "
        "reimbursement, or procurement). Search by topic or issue; use 'all' to retrieve the "
        "complete compact policy set for this case."
    )
    inputs = {
        "topic": {
            "type": "string",
            "description": (
                "Policy topic such as Auslandsreise, private Verlängerung, Doppelfinanzierung, "
                "Kassenbon, Rechnungsempfänger, Rahmenvertrag, or Vergleichsangebote."
            ),
        }
    }
    output_type = "string"

    def __init__(self, policies: dict[str, str]):
        super().__init__()
        self.policies = policies
        self.call_count = 0

    def forward(self, topic: str) -> str:
        self.call_count += 1
        terms = [term.casefold() for term in re.findall(r"\w+", topic) if len(term) > 1]
        if topic.casefold().strip() == "all":
            selected = self.policies
        else:
            ranked = []
            for key, policy in self.policies.items():
                haystack = f"{key} {policy}".casefold()
                score = sum(haystack.count(term) for term in terms)
                if score:
                    ranked.append((score, key, policy))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            selected = {key: policy for _, key, policy in ranked[:4]}
        if not selected:
            return json.dumps(
                {"message": "No direct match. Call lookup_policy with topic='all'.", "policies": {}},
                ensure_ascii=False,
            )
        return json.dumps({"policies": selected}, ensure_ascii=False)


def _parse_date(value: str) -> date:
    for pattern in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date {value!r}; use YYYY-MM-DD or DD.MM.YYYY.")


class CheckFactsTool(Tool):
    name = "check_facts"
    description = (
        "Perform deterministic checks instead of mental arithmetic. The facts object must use one kind: "
        "compare_dates with comparisons [{name,left,operator,right}]; "
        "sum_amounts with amounts, expected, and optional tolerance; "
        "overlap with left and right string lists; or "
        "required_fields with present and required string lists."
    )
    inputs = {
        "facts": {
            "type": "object",
            "description": "Structured check request following one of the schemas in the tool description.",
        }
    }
    output_type = "string"

    def forward(self, facts: dict[str, Any]) -> str:
        kind = facts.get("kind")
        if kind == "compare_dates":
            operators = {
                "<": lambda left, right: left < right,
                "<=": lambda left, right: left <= right,
                "==": lambda left, right: left == right,
                ">=": lambda left, right: left >= right,
                ">": lambda left, right: left > right,
            }
            results = []
            for comparison in facts.get("comparisons", []):
                operator = comparison.get("operator")
                if operator not in operators:
                    raise ValueError(f"Unsupported date operator: {operator}")
                left = _parse_date(str(comparison["left"]))
                right = _parse_date(str(comparison["right"]))
                results.append(
                    {
                        "name": comparison["name"],
                        "passed": operators[operator](left, right),
                        "expression": f"{left.isoformat()} {operator} {right.isoformat()}",
                    }
                )
            if not results:
                raise ValueError("compare_dates requires at least one comparison.")
            return json.dumps({"kind": kind, "results": results})

        if kind == "sum_amounts":
            amounts = [Decimal(str(value)) for value in facts.get("amounts", [])]
            if not amounts:
                raise ValueError("sum_amounts requires at least one amount.")
            expected = Decimal(str(facts["expected"]))
            tolerance = Decimal(str(facts.get("tolerance", "0.01")))
            total = sum(amounts, Decimal("0"))
            return json.dumps(
                {
                    "kind": kind,
                    "total": str(total),
                    "expected": str(expected),
                    "difference": str(total - expected),
                    "passed": abs(total - expected) <= tolerance,
                }
            )

        if kind == "overlap":
            left = {str(value).casefold() for value in facts.get("left", [])}
            right = {str(value).casefold() for value in facts.get("right", [])}
            overlap = sorted(left & right)
            return json.dumps({"kind": kind, "overlap": overlap, "has_overlap": bool(overlap)})

        if kind == "required_fields":
            present = {str(value).casefold() for value in facts.get("present", [])}
            required = {str(value).casefold() for value in facts.get("required", [])}
            missing = sorted(required - present)
            return json.dumps({"kind": kind, "missing": missing, "complete": not missing})

        raise ValueError(
            "facts.kind must be compare_dates, sum_amounts, overlap, or required_fields."
        )


def build_tools(input_root: Path, case_id: str) -> list[Tool]:
    policies = POLICIES_BY_CASE_TYPE[case_type_for(case_id)]
    return [
        log_tool_calls(ListCaseDocumentsTool(input_root, case_id)),
        log_tool_calls(ReadPdfTool(input_root, case_id)),
        log_tool_calls(SearchCaseTool(input_root, case_id)),
        log_tool_calls(LookupPolicyTool(policies)),
        log_tool_calls(CheckFactsTool()),
    ]
