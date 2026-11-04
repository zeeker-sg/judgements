"""Tests for resources.citation_rules (citation linker, step 1: detection).

Example sentences are taken from real eLitigation judgments in the local
cache and fixtures, after Phase 0 text normalisation.
"""

from __future__ import annotations

import pytest

from resources.citation_rules import detect, detect_judgment, detect_parts


def spans(text: str, label: str | None = None):
    return [(m.label, m.text) for m in detect(text) if label is None or m.label == label]


def only(text: str, label: str):
    found = [m for m in detect(text) if m.label == label]
    assert len(found) == 1, found
    return found[0]


# ---------------------------------------------------------------------------
# CITATION
# ---------------------------------------------------------------------------


class TestCitations:
    @pytest.mark.parametrize(
        "text, normalised, kind",
        [
            ("[2026] SGHC 62", "[2026] SGHC 62", "sg_neutral"),
            ("[2025] SGHC(A) 3", "[2025] SGHC(A) 3", "sg_neutral"),
            ("[2020] SGCA(I) 2", "[2020] SGCA(I) 2", "sg_neutral"),
            ("[2024] 5 SLR 194", "[2024] 5 SLR 194", "sg_report"),
            ("[1999] 2 SLR(R) 462", "[1999] 2 SLR(R) 462", "sg_report"),
            ("[1982–1983] SLR 413", "[1982–1983] SLR 413", "sg_report"),
            ("[2002] 1 WLR 1269", "[2002] 1 WLR 1269", "other_report"),
            ("[2009] 3 All ER 404", "[2009] 3 All ER 404", "other_report"),
            ("[1917] P 271", "[1917] P 271", "other_report"),
            ("[2015] EWCA Civ 123", "[2015] EWCA Civ 123", "other_report"),
            ("[2011] EWHC 456 (Ch)", "[2011] EWHC 456 (Ch)", "other_report"),
        ],
    )
    def test_forms(self, text, normalised, kind):
        m = only(f"see {text} for this.", "CITATION")
        assert (m.text, m.normalised, m.kind) == (text, normalised, kind)

    def test_missing_space_before_page_as_in_source(self):
        # eLitigation HTML really has "[2022] 3 SLR</span>924".
        m = only("[2022] 3 SLR924 at [127]", "CITATION")
        assert m.normalised == "[2022] 3 SLR 924"

    def test_revised_edition_is_not_a_citation(self):
        assert spans("the Judicature Act 1969 (2020 Rev Ed)", "CITATION") == []

    @pytest.mark.parametrize(
        "tail, pinpoint",
        [
            (" at [47].", "[47]"),
            (" at [14]–[15]", "[14]–[15]"),
            (" at [38] and [136];", "[38] and [136]"),
            (" (“KS v LHY”) at [21].", "[21]"),
        ],
    )
    def test_pinpoint(self, tail, pinpoint):
        assert only(f"[2024] 5 SLR 194{tail}", "CITATION").pinpoint == pinpoint


# ---------------------------------------------------------------------------
# CASE_NAME
# ---------------------------------------------------------------------------


class TestCaseNames:
    @pytest.mark.parametrize(
        "sentence, name",
        [
            (
                "see Shanmugam Kasiviswanathan v Lee Hsien Yang and another matter [2024] 5 SLR 194",
                "Shanmugam Kasiviswanathan v Lee Hsien Yang and another matter",
            ),
            (
                "see U Myo Nyunt (alias Michael Nyunt) v First Property Holdings Pte Ltd"
                " [2021] 2 SLR 816",
                "U Myo Nyunt (alias Michael Nyunt) v First Property Holdings Pte Ltd",
            ),
            (
                "the court in Qingdao Bohai Construction Group Co, Ltd and others v"
                " Goh Teck Beng and another [2016] 4 SLR 977",
                "Qingdao Bohai Construction Group Co, Ltd and others v Goh Teck Beng and another",
            ),
            ("In Re Nalpon, Zero Geraldo Mario [2018] SGHC 112,", "Re Nalpon, Zero Geraldo Mario"),
            ("relied on R v S [1990] 1 WLR 1", "R v S"),
            (
                "found in R v Civil Service Appeal Board, ex parte Cunningham [1991] 4 All ER 310",
                "R v Civil Service Appeal Board, ex parte Cunningham",
            ),
            # The court that decided it is not part of the name.
            (
                "the Singapore High Court in Qingdao Bohai Construction Group Co, Ltd v"
                " Goh Teck Beng [2016] 4 SLR 977",
                "Qingdao Bohai Construction Group Co, Ltd v Goh Teck Beng",
            ),
            (
                "Chua Lee Ming J in Youprint Productions Pte Ltd v Mak Sook Ling [2023] 3 SLR 1130",
                "Youprint Productions Pte Ltd v Mak Sook Ling",
            ),
            ("as in de Dampierre v de Dampierre [1988] AC 92", "de Dampierre v de Dampierre"),
        ],
    )
    def test_name_before_citation(self, sentence, name):
        assert only(sentence, "CASE_NAME").text == name

    def test_list_of_cases_split_at_comma_after_page(self):
        names = spans("see A Pte Ltd v B [2010] 1 SLR 1, C v D [2011] 2 SLR 2.", "CASE_NAME")
        assert [t for _, t in names] == ["A Pte Ltd v B", "C v D"]

    def test_no_case_name_without_v_or_re(self):
        assert spans("as decided in [2026] SGHC 62 at [3]", "CASE_NAME") == []


# ---------------------------------------------------------------------------
# SHORT_NAME
# ---------------------------------------------------------------------------


class TestShortNames:
    def test_defined_after_citation(self):
        m = only("Goh Teck Beng [2016] 4 SLR 977 (“Qingdao Bohai”) at [37]", "SHORT_NAME")
        assert m.text == "Qingdao Bohai"

    def test_defined_after_pinpoint(self):
        m = only("v Mak Sook Ling [2023] 3 SLR 1130 at [12] (“Youprint”)", "SHORT_NAME")
        assert m.text == "Youprint"

    def test_defined_after_statute(self):
        m = only("the Criminal Procedure Code 2010 (2020 Rev Ed) (“CPC”), and", "SHORT_NAME")
        assert m.text == "CPC"

    def test_other_definitions_are_ignored(self):
        # Most bracketed definitions name documents or proceedings, not cases.
        text = "an action in DC/OC 1154/2025 (“OC 1154”) and (the “Default Judgment”)"
        assert spans(text, "SHORT_NAME") == []


# ---------------------------------------------------------------------------
# INSTRUMENT and PROVISION
# ---------------------------------------------------------------------------


class TestStatutes:
    @pytest.mark.parametrize(
        "sentence, title",
        [
            ("under the Defamation Act 1957 (2020 Rev Ed),", "Defamation Act 1957 (2020 Rev Ed)"),
            ("s 109 of the Penal Code (Cap 224, 1985 Rev Ed)", "Penal Code (Cap 224, 1985 Rev Ed)"),
            ("Under the Moneylenders Act 2008, a loan", "Moneylenders Act 2008"),
            ("the Stamp Duties Act requires", "Stamp Duties Act"),
            ("O 3 r 2(8) of the Rules of Court 2021", "Rules of Court 2021"),
            (
                "s 11A of the Women’s Charter 1961 (2020 Rev Ed)",
                "Women’s Charter 1961 (2020 Rev Ed)",
            ),
            ("Art 26 of the Energy Charter Treaty.", "Energy Charter Treaty"),
            ("Art 4 of the First Schedule of the Stamp Duties Act", "Stamp Duties Act"),
        ],
    )
    def test_instrument(self, sentence, title):
        assert only(sentence, "INSTRUMENT").text == title

    @pytest.mark.parametrize(
        "sentence",
        ["The Act says so.", "What Can They Act about it?", "This Code applies."],
    )
    def test_not_an_instrument(self, sentence):
        assert spans(sentence, "INSTRUMENT") == []

    @pytest.mark.parametrize(
        "sentence, provision",
        [
            ("under s 52(14) of the AMLA", "s 52(14)"),
            ("s 17A(2)–(3) of the SCJA", "s 17A(2)–(3)"),
            ("ss 4 and 11 of the Act", "ss 4 and 11"),
            ("O 3 r 2(8) of the Rules", "O 3 r 2(8)"),
            ("section 11A of the Charter", "section 11A"),
            ("Art 54(1) of the Convention", "Art 54(1)"),
            ("s 5(1)(a) read with", "s 5(1)(a)"),
        ],
    )
    def test_provision(self, sentence, provision):
        assert only(sentence, "PROVISION").text == provision

    def test_possessive_s_before_a_number_is_not_a_provision(self):
        assert spans("the Respondent’s 25% contribution", "PROVISION") == []

    def test_judgment_paragraph_reference_is_not_a_provision(self):
        assert spans("Statement of claim at para 1.", "PROVISION") == []


def test_mentions_are_sorted_and_do_not_overlap():
    text = (
        "see Tan Ah Thee v Lim Soo Foong [2009] 3 SLR(R) 957 at [12] (“Tan Ah Thee”)"
        " and s 3 of the State Immunity Act 1979 (2020 Rev Ed)."
    )
    found = detect(text)
    assert [m.start for m in found] == sorted(m.start for m in found)
    for a, b in zip(found, found[1:]):
        assert a.end <= b.start
    assert [m.label for m in found] == [
        "CASE_NAME",
        "CITATION",
        "SHORT_NAME",
        "PROVISION",
        "INSTRUMENT",
    ]


# ---------------------------------------------------------------------------
# Labelling guide decisions (design doc, "Labelling guide" tab)
# ---------------------------------------------------------------------------


class TestGuideDecisions:
    @pytest.mark.parametrize(
        "sentence, number",
        [
            ("his appeal in CA/CCA 20/2022 was dismissed", "CA/CCA 20/2022"),
            ("a defamation action in DC/OC 1154/2025 (“OC 1154”)", "DC/OC 1154/2025"),
            ("the costs of HC/SUM 1629/2025", "HC/SUM 1629/2025"),
            ("the costs order for SUM 1371/2025", "SUM 1371/2025"),
            ("In Suit No 123 of 2019, the plaintiff", "Suit No 123 of 2019"),
            ("Magistrate’s Appeal No 9091 of 2015", "Magistrate’s Appeal No 9091 of 2015"),
        ],
    )
    def test_case_number(self, sentence, number):
        assert only(sentence, "CASE_NUMBER").text == number

    def test_act_number_is_not_a_case_number(self):
        assert spans("amended by Act No 5 of 2010", "CASE_NUMBER") == []

    def test_short_name_after_case_number_is_not_labelled(self):
        # Decision 7: "OC 1154" names the proceeding, not a case or Act.
        assert spans("an action in DC/OC 1154/2025 (“OC 1154”)", "SHORT_NAME") == []

    @pytest.mark.parametrize(
        "text", ["(2019) 31 SAcLJ 123", "[2015] SJLS 45", "(2012) 128 LQR 1", "(2010) 22 SAcLJ 1"]
    )
    def test_journal_articles_are_not_citations(self, text):
        assert spans(f"see Tan, {text} at 5.", "CITATION") == []

    def test_malayan_law_journal_is_a_report(self):
        assert only("see [1985] 1 MLJ 123.", "CITATION").kind == "other_report"

    @pytest.mark.parametrize(
        "sentence, provision",
        [
            ("under para 4 of the First Schedule to the Act", "para 4"),
            ("paras 4 and 5 of the Second Schedule", "paras 4 and 5"),
            ("para 2(a) of Schedule 2", "para 2(a)"),
        ],
    )
    def test_schedule_paragraph_is_a_provision(self, sentence, provision):
        assert only(sentence, "PROVISION").text == provision


class TestGoldSampleFixes:
    """False positives and cut-offs found in the 200-judgment gold sample."""

    @pytest.mark.parametrize(
        "sentence, provision",
        [("(“CPR”) r 17.4 gave effect", "r 17.4"), ("Art 26.1 of the Convention", "Art 26.1")],
    )
    def test_dotted_provision_numbers(self, sentence, provision):
        assert only(sentence, "PROVISION").text == provision

    def test_charterer_is_not_a_charter(self):
        text = "after service of the Enforcement Order and the Charterer not having paid"
        assert spans(text, "INSTRUMENT") == []

    def test_contract_number_is_not_a_case_number(self):
        assert spans("(Contract No GB013/2021P), under which", "CASE_NUMBER") == []


class TestCaseFileDocuments:
    """Decision 14: contracts, pleadings and affidavits are not instruments."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "Art 26.1 of the Contract",
            "Art 3 of the Charterparty",
            "Art 4 of the Charter Party dated 1 May",
            "r 5 of the Sale and Purchase Agreement",
            "s 11 of the Policy",
            # Pleadings, affidavits and submissions (also decision 14).
            "para 12 of the Statement of Claim",
            "paras 5-7 of the Reply",
            "para 20 of the 1st Affidavit of Tan Ah Kow dated 3 May 2024",
            "at para 8 of his AEIC",
            "the Plaintiff’s Closing Submissions at para 45",
        ],
    )
    def test_document_paragraph_is_not_a_provision(self, sentence):
        assert spans(sentence, "PROVISION") == []

    @pytest.mark.parametrize(
        "sentence, provision",
        [
            ("s 3 of the Unfair Contract Terms Act 1977", "s 3"),
            ("under r 17.4 of the SIAC Rules", "r 17.4"),
            ("Art 7 of the Charter", "Art 7"),
            ("s 2 of the Act defines Agreement", "s 2"),
        ],
    )
    def test_instrument_provisions_still_count(self, sentence, provision):
        assert only(sentence, "PROVISION").text == provision


class TestDocumentsNamedAfterProvisions:
    """Decision 15: "section 122(6) statement" names a document, not the section."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "in his section 122(6) statement, the accused",
            "the s 23 statement recorded on 3 May",
            "his s 22 statements",
        ],
    )
    def test_named_document_is_not_a_provision(self, sentence):
        assert spans(sentence, "PROVISION") == []

    def test_section_itself_is_still_a_provision(self):
        assert only("the cautioned statement under s 23 of the CPC", "PROVISION").text == "s 23"


class TestGoldBatchOne:
    """Fixes from scoring the first 98 labelled gold passages (rules-4)."""

    @pytest.mark.parametrize(
        "text, citation",
        [
            ("Mills v Mills (1937-38) 60 CLR 150, quoted", "(1937-38) 60 CLR 150"),
            ("reproduced in [1843-60] All ER 249 (HL)", "[1843-60] All ER 249"),
        ],
    )
    def test_year_range_with_two_digit_end(self, text, citation):
        assert only(text, "CITATION").text == citation

    @pytest.mark.parametrize(
        "text, number",
        [
            ("Case Number : CA 83/2002/A, Decision", "CA 83/2002/A"),
            ("Case Number : CA 83/2002/A, Suit 755/2001/A", "Suit 755/2001/A"),
            (
                "who instituted Admiralty in Rem No 181 of 2009 against",
                "Admiralty in Rem No 181 of 2009",
            ),
        ],
    )
    def test_case_numbers(self, text, number):
        assert number in [t for _, t in spans(text, "CASE_NUMBER")]

    def test_gazette_number_is_not_a_case_number(self):
        assert spans("Rules 1963 (GN No S 98/1963) (at [41])", "CASE_NUMBER") == []

    @pytest.mark.parametrize(
        "text, name",
        [
            (
                "in BTI 2014 LLC v Sequana SA and others [2022] UKSC 25 (“Sequana”)",
                "BTI 2014 LLC v Sequana SA and others",
            ),
            (
                "decision: Carl Zeiss Stiftung v Rayner & Keeler Ltd (No 2) [1967] 1 AC 853",
                "Carl Zeiss Stiftung v Rayner & Keeler Ltd (No 2)",
            ),
            (
                "Finally, Re Lim Chor Pee, ex parte Law Society of Singapore [1985-1986] SLR(R) 226",
                "Re Lim Chor Pee, ex parte Law Society of Singapore",
            ),
            (
                "argument – How Weng Fun and ors v Sengkang Town Council and other appeals [2023] SGCA 21",
                "How Weng Fun and ors v Sengkang Town Council and other appeals",
            ),
            ("In The “Jian He” [1993] 3 SLR(R) 432 (“Jian He”)", "The “Jian He”"),
            (
                "(see Natixis, Singapore Branch v Lim Oon Kuin and others [2024] 3 SLR 1502",
                "Natixis, Singapore Branch v Lim Oon Kuin and others",
            ),
        ],
    )
    def test_case_names_before_citation(self, text, name):
        assert only(text, "CASE_NAME").text == name

    def test_case_names_without_citation(self):
        text = (
            "Ms Lim attempted to distinguish McDonald’s Hamburgers Limited v Burgerking (UK) "
            "Limited and Fraser & Neave Ltd v Yeo Hiap Seng Ltd as passing off cases. In "
            "Fraser & Neave Ltd v Yeo Hiap Seng Ltd, the court decided"
        )
        assert [t for _, t in spans(text, "CASE_NAME")] == [
            "McDonald’s Hamburgers Limited v Burgerking (UK) Limited",
            "Fraser & Neave Ltd v Yeo Hiap Seng Ltd",
            "Fraser & Neave Ltd v Yeo Hiap Seng Ltd",
        ]

    @pytest.mark.parametrize(
        "text, title",
        [
            (
                "under r 10 of the Advocates and Solicitors (Disciplinary Proceedings) Rules 1963 (GN",
                "Advocates and Solicitors (Disciplinary Proceedings) Rules 1963",
            ),
            (
                "that the Administration of Justice (Protection) Act 2016 (“AJPA”) exists",
                "Administration of Justice (Protection) Act 2016",
            ),
        ],
    )
    def test_instrument_with_bracketed_word(self, text, title):
        assert only(text, "INSTRUMENT").text == title

    def test_long_title_quote_is_not_an_instrument(self):
        assert spans("The words “An Act to provide for monogamous marriages”", "INSTRUMENT") == []

    @pytest.mark.parametrize(
        "text, provisions",
        [
            ("the matter. Section 93(1) of the Act provided", ["Section 93(1)"]),
            (
                "a laboratory test (s 70(1)(b), read with s 71(1), of the RTA).",
                ["s 70(1)(b)", "s 71(1)"],
            ),
            ("under s 22. Order 89B r 2(2) of the Rules of Court", ["s 22", "Order 89B r 2(2)"]),
            ("3-(1) Subject to subsection (2), any person", ["subsection (2)"]),
            ("the relevant date. Subsection (5) of s 33 sets out", ["Subsection (5) of s 33"]),
            ("for the present purposes, para (e) was germane", ["para (e)"]),
            ("Per ss 46(1)(a) or 46(1)(b) of the GIA, these", ["ss 46(1)(a) or 46(1)(b)"]),
        ],
    )
    def test_provisions(self, text, provisions):
        assert [t for _, t in spans(text, "PROVISION")] == provisions

    def test_named_statement_with_act_abbreviation(self):
        text = "when the s 122(6) CPC statement was being taken"
        assert spans(text, "PROVISION") == [] and spans(text, "SHORT_NAME") == []


class TestShortNamesInJudgment:
    """Abbreviations and short names used away from their definition."""

    @pytest.mark.parametrize(
        "text, name",
        [
            ("It follows that s 52 of the TCA would avail them", "TCA"),
            ("Section 32(1) of the EA expressly contemplates", "EA"),
            ("did not invoke section 9 of the Act to deem", "Act"),
        ],
    )
    def test_abbreviation_after_provision(self, text, name):
        assert only(text, "SHORT_NAME").text == name

    def test_act_without_provision_or_definition_is_not_labelled(self):
        assert spans("the Act was amended in 2008", "SHORT_NAME") == []

    def test_supra_reference(self):
        assert only("In Bayoil ([36] supra), the debtor did not", "SHORT_NAME").text == "Bayoil"

    @pytest.mark.parametrize(
        "text",
        [
            "on the commencement of the Arbitration (see [21] above)",
            "conveyed informally through Ms How ([318] above)",
        ],
    )
    def test_cross_reference_is_not_a_short_name(self, text):
        assert spans(text, "SHORT_NAME") == []

    def test_names_carry_across_a_judgment(self):
        first, second = detect_judgment(
            [
                "the Electronic Transactions Act 2010 (“ETA”) and Goh Teck Beng v Tan "
                "[2016] 4 SLR 977 (“Qingdao Bohai”) at [37]",
                "Under the ETA, as Qingdao Bohai at [40] shows, the Act was",
            ]
        )
        assert [(m.label, m.text, m.kind) for m in second] == [
            ("SHORT_NAME", "ETA", "use"),
            ("SHORT_NAME", "Qingdao Bohai", "use"),
        ]

    def test_use_never_splits_an_instrument(self):
        first, second = detect_judgment(
            ["s 5 of the Act applies", "the Misuse of Drugs Act 1973 (2020 Rev Ed) applies"]
        )
        assert [m.label for m in second] == ["INSTRUMENT"]

    def test_name_defined_with_the_needs_the(self):
        first, second = detect_judgment(
            [
                "O 21 r 22(2) of the Rules of Court 2021 (“the Rules”). When costs",
                "Under the Rules, costs follow; the Supreme Court Rules differ",
            ]
        )
        assert [(m.text, m.kind) for m in first if m.label == "SHORT_NAME"] == [
            ("Rules", "defined")
        ]
        assert [(m.text, m.kind) for m in second if m.label == "SHORT_NAME"] == [("Rules", "use")]

    def test_use_does_not_run_into_a_longer_name(self):
        first, second = detect_judgment(["s 2 of the CYP applies", "the CYP(A)A 2019 applies"])
        assert [m for m in second if m.label == "SHORT_NAME"] == []

    def test_party_after_provision_is_not_a_short_name(self):
        assert spans("the inquiry under s 33B(2)(b) in the PP", "SHORT_NAME") == []

    def test_detect_parts_reads_footnotes_after_their_paragraph(self):
        fragments = [
            {
                "id": "j_0001",
                "content_text": "the Evidence Act 1893 (“EA”) applies",
                "footnote_text": '["s 32 of the EA"]',
            },
            {"id": "j_0002", "content_text": "Under the EA, it is", "footnote_text": None},
        ]
        parts = detect_parts(fragments)
        assert list(parts) == [
            ("j_0001", "content"),
            ("j_0001", "footnote 1"),
            ("j_0002", "content"),
        ]
        text, mentions = parts[("j_0002", "content")]
        assert [(m.text, m.kind) for m in mentions] == [("EA", "use")]
