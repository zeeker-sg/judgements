"""Tests for linking case citations to ``judgments`` rows."""

import pytest

from resources.citation_linker import (
    MATCH_THRESHOLD,
    Catalog,
    Context,
    break_tie_by_cues,
    break_tie_by_pinpoint,
    break_tie_by_text,
    cited_names,
    link_mention,
    link_neutral,
    link_report,
    name_score,
    name_tokens,
)
from resources.citation_rules import detect


def row(id, citation, case_name, date, court=""):
    return {
        "id": id,
        "citation": citation,
        "case_name": case_name,
        "decision_date": date,
        "court": court or citation.split()[1].replace("(", "").replace(")", ""),
    }


CATALOG = Catalog(
    [
        row(
            "smile_hc",
            "[2011] SGHC 266",
            "Smile Inc Dental Surgeons Pte Ltd v Lui Andrew Stewart",
            "2011-12-16",
        ),
        row(
            "smile_ca",
            "[2012] SGCA 39",
            "Smile Inc Dental Surgeons Pte Ltd v Lui Andrew Stewart",
            "2012-07-31",
        ),
        row("kho", "[2016] SGCA 21", "JABING KHO v PUBLIC PROSECUTOR", "2016-04-05"),
        row("nf", "[2006] SGHC 165", "Public Prosecutor v NF", "2006-09-21"),
        row("tan", "[2006] SGHC 166", "Public Prosecutor v Tan Ah Kow", "2006-09-22"),
        row("htsrl", "[2016] SGHC 15", "HT S.R.L. v Wee Shuo Woon", "2016-02-16"),
        row("re_ag", "[2023] SGHC 78", "Re: THE ATTORNEY-GENERAL", "2023-03-01"),
        row("abdullah", "[2017] SGCA 4", "MUHAMMAD BIN ABDULLAH v PUBLIC PROSECUTOR", "2017-01-12"),
        row("nur", "[2017] SGCA 66", "MUHAMMAD NUR BIN ABDULLAH v PUBLIC PROSECUTOR", "2017-11-20"),
        row(
            "lcw_hc",
            "[2006] SGHC 116",
            "Lee Chee Wei v Tan Hor Peow Victor and Others",
            "2006-06-30",
        ),
        row(
            "lcw_ca",
            "[2007] SGCA 22",
            "Lee Chee Wei v Tan Hor Peow Victor and Others and Another Appeal",
            "2007-04-16",
        ),
        row("dup1", "[2015] SGHCR 2", "Lee Seng Eder v Wee Kim Chwee and others", "2015-01-06"),
        row("dup2", "[2015] SGHCR 2", "Lee Seng Eder v Wee Kim Chwee and others", "2015-01-06"),
        row("nospace", "[2013] SGHC115", "Public Prosecutor v Charan Singh", "2013-05-27"),
        row(
            "progen_hc",
            "[2009] SGHC 286",
            "Liquidators of Progen Engineering Pte Ltd v Progen Holdings Ltd",
            "2009-12-23",
        ),
        row(
            "progen_ca",
            "[2010] SGCA 31",
            "Chee Yoh Chuang and Another (as Liquidators of Progen Engineering Pte Ltd"
            " (In Liquidation)) v Progen Holdings Ltd",
            "2010-08-31",
        ),
        row("old", "[2000] SGHC 1", "Alpha Pte Ltd v Beta Pte Ltd", "2000-04-28"),
    ]
)


class TestNames:
    def test_pp_is_public_prosecutor(self):
        assert name_tokens("PP v NF") == name_tokens("Public Prosecutor v NF")

    def test_company_forms_and_joinder_drop_out(self):
        assert name_tokens("QINGDAO BOHAI CONSTRUCTION GROUP CO., LTD & 4 Ors") == {
            "qingdao",
            "bohai",
            "construction",
            "group",
        }

    def test_dotted_abbreviations_close_up(self):
        assert "srl" in name_tokens("HT S.R.L. v Wee Shuo Woon")

    def test_capitals_match(self):
        assert name_score("Pram Nair v Public Prosecutor", "PRAM NAIR v PUBLIC PROSECUTOR") == 1

    def test_each_side_must_share_a_word(self):
        assert name_score("Public Prosecutor v Tan Ah Kow", "Public Prosecutor v NF") == 0

    def test_a_shared_word_on_each_side_is_enough(self):
        # Parties swapped between courts still match when a word is common to both.
        assert (
            name_score(
                "Glaziers Engineering v WCS Engineering", "WCS Engineering v Glaziers Engineering"
            )
            > 0
        )

    def test_swapped_parties_do_not_match(self):
        # Known limit: the appeal's name often lists the parties the other way
        # round ("PP v Beh Chew Boo" below, "Beh Chew Boo v PP" on appeal).
        assert (
            name_score("Beh Chew Boo v Public Prosecutor", "PUBLIC PROSECUTOR v BEH CHEW BOO") == 0
        )

    def test_short_catalog_name_does_not_swallow_a_long_one(self):
        score = name_score(
            "Masoud Rahimi bin Mehrzad v Attorney-General", "Re: THE ATTORNEY-GENERAL"
        )
        assert score < MATCH_THRESHOLD


class TestNeutral:
    def test_exact(self):
        link = link_neutral(CATALOG, "[2016] SGCA 21")
        assert (link.tier, link.target_id) == ("exact", "kho")

    def test_spacing_ignored(self):
        assert link_neutral(CATALOG, "[2013] SGHC 115").target_id == "nospace"

    def test_missing_is_none(self):
        assert link_neutral(CATALOG, "[2019] SGFC 6").tier == "none"

    def test_listed_twice_is_tied(self):
        link = link_neutral(CATALOG, "[2015] SGHCR 2")
        assert link.tier == "tied" and not link.linked


class TestReport:
    def test_unique_by_name_and_year(self):
        link = link_report(CATALOG, "PP v NF", 2006)
        assert (link.tier, link.target_id) == ("unique", "nf")

    def test_year_before_allowed(self):
        assert link_report(CATALOG, "HT SRL v Wee Shuo Woon", 2017).target_id == "htsrl"

    def test_two_years_before_not_allowed(self):
        assert link_report(CATALOG, "HT SRL v Wee Shuo Woon", 2018).tier == "none"

    def test_before_catalog_starts(self):
        assert (
            link_report(CATALOG, "Fraser & Neave Ltd v Yeo Hiap Seng Ltd", 1988).tier == "too_early"
        )

    def test_no_name(self):
        assert link_report(CATALOG, None, 2016).tier == "none"

    def test_weak_match_not_linked(self):
        link = link_report(CATALOG, "Masoud Rahimi bin Mehrzad v Attorney-General", 2023)
        assert link.tier == "weak" and link.target_id is None

    def test_high_court_and_appeal_tie(self):
        link = link_report(CATALOG, "Smile Inc Dental Surgeons Pte Ltd v Lui Andrew Stewart", 2012)
        assert link.tier == "tied" and {c for c, _ in link.candidates} == {"smile_hc", "smile_ca"}

    def test_citing_judgment_and_later_ones_left_out(self):
        name = "Smile Inc Dental Surgeons Pte Ltd v Lui Andrew Stewart"
        assert link_report(CATALOG, name, 2012, citing_id="smile_ca").target_id == "smile_hc"
        assert link_report(CATALOG, name, 2012, citing_date="2012-01-01").target_id == "smile_hc"

    def test_fewer_extra_words_wins(self):
        link = link_report(CATALOG, "Muhammad bin Abdullah v Public Prosecutor", 2017)
        assert link.target_id == "abdullah"

    def test_fewer_extra_words_never_picks_between_trial_and_appeal(self):
        # [2010] 4 SLR 1089 reports the appeal under the trial's name.
        name = "Liquidators of Progen Engineering Pte Ltd v Progen Holdings Ltd"
        link = link_report(CATALOG, name, 2010)
        assert link.tier == "tied" and {c for c, _ in link.candidates} == {
            "progen_hc",
            "progen_ca",
        }

    def test_another_appeal_picks_appellate_court(self):
        name = "Lee Chee Wei v Tan Hor Peow Victor and others and another appeal"
        link = link_report(CATALOG, name, 2007)
        assert (link.tier, link.target_id) == ("resolved", "lcw_ca")


SMILE = "Smile Inc Dental Surgeons Pte Ltd v Lui Andrew Stewart"


def tie(citation, texts):
    link = link_report(CATALOG, SMILE, int(citation[1:5]))
    return break_tie_by_text(CATALOG, link, citation, SMILE, texts.get)


class TestTieByText:
    def test_later_judgment_cites_this_report(self):
        texts = {"smile_ca": [f"The Judge's decision in {SMILE} [2012] 1 SLR 847 at [12]."]}
        link = tie("[2012] 1 SLR 847", texts)
        assert (link.tier, link.target_id) == ("resolved", "smile_hc")

    def test_later_judgment_cites_the_other_report(self):
        texts = {"smile_ca": [f"The Judge's decision in {SMILE} [2012] 1 SLR 847 at [12]."]}
        link = tie("[2012] 4 SLR 308", texts)
        assert (link.tier, link.target_id) == ("resolved", "smile_ca")

    def test_older_report_in_the_same_dispute_proves_nothing(self):
        texts = {"smile_ca": [f"An earlier round: {SMILE} [2009] 2 SLR(R) 10."]}
        assert tie("[2012] 4 SLR 308", texts).tier == "tied"

    def test_earlier_judgment_text_ignored(self):
        texts = {"smile_hc": [f"See {SMILE} [2012] 1 SLR 847."]}
        assert tie("[2012] 4 SLR 308", texts).tier == "tied"

    def test_no_text(self):
        assert tie("[2012] 4 SLR 308", {}).tier == "tied"


def ctx(text, citing_court="SGHC", citation="[2012] 4 SLR 308"):
    return Context(
        text, text.index(SMILE), text.index(citation) + len(citation), SMILE, citing_court
    )


def cue(text, citing_court="SGHC"):
    link = link_report(CATALOG, SMILE, 2012)
    return break_tie_by_cues(CATALOG, link, ctx(text, citing_court))


class TestTieByCues:
    def test_court_of_appeal_named_before(self):
        link = cue(f"As the Court of Appeal held in {SMILE} [2012] 4 SLR 308, it is so.")
        assert (link.tier, link.target_id) == ("resolved", "smile_ca")

    def test_court_named_after(self):
        link = cue(f"In {SMILE} [2012] 4 SLR 308 (“Smile”), the Court of Appeal held that")
        assert link.target_id == "smile_ca"

    def test_this_court_is_the_citing_courts_level(self):
        text = f"See the decision of this court in {SMILE} [2012] 4 SLR 308 at [5]."
        assert cue(text, citing_court="SGCA").target_id == "smile_ca"
        assert cue(text, citing_court="SGHC").target_id == "smile_hc"

    def test_our_remarks(self):
        text = f"The Judge relied on our remarks in {SMILE} [2012] 4 SLR 308."
        assert cue(text, citing_court="SGCA").target_id == "smile_ca"

    def test_affirmed_on_appeal_means_the_trial(self):
        link = cue(f"See {SMILE} [2012] 4 SLR 308, affirmed on appeal.")
        assert link.target_id == "smile_hc"

    def test_decision_date(self):
        link = cue(f"The claim was dismissed on 16 December 2011: {SMILE} [2012] 4 SLR 308.")
        assert (link.target_id, link.note) == ("smile_hc", "text gives the date 2011-12-16")

    def test_court_words_not_next_to_the_name_prove_nothing(self):
        text = f"The High Court Judge relied on {SMILE} [2012] 4 SLR 308."
        assert cue(text).tier == "tied"

    def test_no_cue(self):
        assert cue(f"See {SMILE} [2012] 4 SLR 308 at [12].").tier == "tied"


PARAS = {
    "smile_hc": {n: "an unrelated point about costs" for n in range(1, 11)},
    "smile_ca": {n: "an unrelated point about costs" for n in range(1, 61)},
}


def pinpoint(text, paras=PARAS):
    link = link_report(CATALOG, SMILE, 2012)
    return break_tie_by_pinpoint(CATALOG, link, ctx(text), paras.get)


class TestTieByPinpoint:
    def test_candidate_too_short_for_the_pinpoint(self):
        link = pinpoint(f"See {SMILE} [2012] 4 SLR 308 at [40].")
        assert (link.tier, link.target_id) == ("resolved", "smile_ca")

    def test_paragraph_that_matches_the_citing_text(self):
        paras = {k: dict(v) for k, v in PARAS.items()}
        paras["smile_ca"][5] = "a restraint of trade clause binding a dentist is unenforceable"
        text = f"A restraint of trade clause binding a dentist is unenforceable: {SMILE} [2012] 4 SLR 308 at [5]."
        link = pinpoint(text, paras)
        assert link.target_id == "smile_ca" and link.note.startswith("[5] matches")

    def test_paragraphs_alike_stay_tied(self):
        assert pinpoint(f"On costs, see {SMILE} [2012] 4 SLR 308 at [5].").tier == "tied"

    def test_unknown_text_stays_tied(self):
        paras = {"smile_hc": None, "smile_ca": PARAS["smile_ca"]}
        assert pinpoint(f"See {SMILE} [2012] 4 SLR 308 at [5].", paras).tier == "tied"

    def test_no_pinpoint(self):
        assert pinpoint(f"See {SMILE} [2012] 4 SLR 308.").tier == "tied"


class TestPairing:
    def pairs(self, text):
        mentions = detect(text)
        return {mentions[i].text: name for i, name in cited_names(mentions, text).items()}

    def test_name_before_citation(self):
        assert self.pairs("See Kho Jabing v PP [2016] 3 SLR 135 at [5].") == {
            "[2016] 3 SLR 135": "Kho Jabing v PP"
        }

    def test_name_carries_over_a_chain(self):
        pairs = self.pairs("In Kho Jabing v PP [2016] SGCA 21; [2016] 3 SLR 135, the court held.")
        assert pairs["[2016] 3 SLR 135"] == "Kho Jabing v PP"

    def test_other_words_break_the_pairing(self):
        pairs = self.pairs("In Kho Jabing v PP the court held, as in [2016] 3 SLR 135, that.")
        assert "[2016] 3 SLR 135" not in pairs


def test_link_mention_skips_foreign_reports():
    m = next(
        x
        for x in detect("Bradford & Bingley Building Society v Seddon [1999] 1 WLR 1482")
        if x.label == "CITATION"
    )
    assert link_mention(CATALOG, m) is None


@pytest.mark.parametrize("citation", ["[2016] SGCA 21", "[2016]  SGCA  21"])
def test_link_mention_neutral(citation):
    m = next(x for x in detect(f"See {citation}.") if x.label == "CITATION")
    assert link_mention(CATALOG, m).target_id == "kho"
