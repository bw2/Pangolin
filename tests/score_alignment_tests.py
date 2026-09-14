"""Tests for pangolin/score_alignment.py.

These need only numpy, not torch or the models.

Run from the repo root with:  python3 -m unittest tests.score_alignment_tests -v
"""

import unittest

import numpy as np

from pangolin.score_alignment import (
    align_ref_and_alt_scores, ref_and_alt_bases_for_position, trim_shared_bases)

# A distance of 5, so the window holds 2*D + len(ref) positions and the variant's first base is at index D.
D = 5


def scores(length, first):
    """Distinct increasing scores, so a moved, collapsed or copied value shows up in the result."""
    return np.arange(first, first + length, dtype=float)


def legacy_alt_scores(alt_scores, ref, alt, d):
    """Pangolin's realignment before score_alignment.py, kept to pin the spellings it already got right."""
    l = 2*d+1
    ndiff = abs(len(ref)-len(alt))
    if len(ref) > len(alt):
        return np.concatenate([alt_scores[0:l//2+1], np.zeros(ndiff), alt_scores[l//2+1:]])
    if len(ref) < len(alt):
        return np.concatenate([
            alt_scores[0:l//2],
            np.max(alt_scores[l//2:l//2+ndiff+1], keepdims=True),
            alt_scores[l//2+ndiff+1:]])
    return alt_scores


class TrimSharedBasesTest(unittest.TestCase):

    def test_snv_is_unchanged(self):
        self.assertEqual(trim_shared_bases("G", "A"), (0, "G", "A"))

    def test_unchanged_bases_before_the_change_are_trimmed(self):
        self.assertEqual(trim_shared_bases("TG", "TA"), (1, "G", "A"))
        self.assertEqual(trim_shared_bases("GCTG", "GCTA"), (3, "G", "A"))

    def test_unchanged_bases_after_the_change_are_trimmed(self):
        self.assertEqual(trim_shared_bases("ACT", "CCT"), (0, "A", "C"))
        self.assertEqual(trim_shared_bases("CGA", "TGA"), (0, "C", "T"))

    def test_genuine_mnv_is_unchanged(self):
        self.assertEqual(trim_shared_bases("TG", "CA"), (0, "TG", "CA"))

    def test_indels_with_one_anchor_base_are_unchanged(self):
        self.assertEqual(trim_shared_bases("GGGC", "G"), (0, "GGGC", "G"))
        self.assertEqual(trim_shared_bases("A", "AGAGAG"), (0, "A", "AGAGAG"))

    def test_indels_with_extra_shared_bases_keep_one_anchor_base(self):
        self.assertEqual(trim_shared_bases("CTG", "CT"), (1, "TG", "T"))
        self.assertEqual(trim_shared_bases("CA", "CAT"), (1, "A", "AT"))

    def test_shared_bases_at_the_end_are_trimmed_before_those_at_the_start(self):
        # CAA>CA could become CA>C or, one base later, AA>A; trimming the end first gives CA>C
        self.assertEqual(trim_shared_bases("CAA", "CA"), (0, "CA", "C"))

    def test_deletion_insertion_trims_to_the_bases_it_changes(self):
        self.assertEqual(trim_shared_bases("CAT", "CGGT"), (1, "A", "GG"))

    def test_comparison_ignores_case(self):
        self.assertEqual(trim_shared_bases("TG", "tA"), (1, "G", "A"))


class AlignRefAndAltScoresTest(unittest.TestCase):

    def align(self, ref, alt):
        """Return (ref_scores, alt_scores, aligned_ref, aligned_alt).

        REF scores start at 1 and ALT scores at 1000, so every value shows which track it came from.
        """
        ref_scores = scores(2*D + len(ref), first=1)
        alt_scores = scores(2*D + len(alt), first=1000)
        aligned_ref, aligned_alt = align_ref_and_alt_scores(ref_scores, alt_scores, ref, alt, D)
        return ref_scores, alt_scores, aligned_ref, aligned_alt

    # --- the spellings Pangolin already supported keep scoring exactly as they did ---

    def test_same_length_alleles_are_not_realigned(self):
        for ref, alt in (("G", "A"), ("TG", "TA"), ("GCTG", "GCTA"), ("TG", "CA"), ("ACT", "CCT")):
            with self.subTest(ref=ref, alt=alt):
                ref_scores, alt_scores, aligned_ref, aligned_alt = self.align(ref, alt)
                np.testing.assert_array_equal(aligned_ref, ref_scores)
                np.testing.assert_array_equal(aligned_alt, alt_scores)

    def test_variants_written_without_shared_bases_match_the_original_realignment(self):
        for ref, alt in (("G", "A"), ("TG", "CA"), ("GGGC", "G"), ("A", "AGAGAG")):
            with self.subTest(ref=ref, alt=alt):
                ref_scores, alt_scores, aligned_ref, aligned_alt = self.align(ref, alt)
                np.testing.assert_array_equal(aligned_ref, ref_scores)
                np.testing.assert_array_equal(aligned_alt, legacy_alt_scores(alt_scores, ref, alt, D))

    # --- a variant padded with shared bases is now realigned where its alleles differ ---

    def test_padded_deletion_is_realigned_where_the_alleles_differ(self):
        # CTG>CT is TG>T one base later: the T keeps its score and only the deleted G gets zero
        ref_scores, alt_scores, aligned_ref, aligned_alt = self.align("CTG", "CT")
        np.testing.assert_array_equal(aligned_ref, ref_scores)
        np.testing.assert_array_equal(aligned_alt, np.concatenate(
            [alt_scores[:D+2], np.zeros(1), alt_scores[D+2:]]))

    def test_the_old_formula_would_have_mis_scored_a_padded_deletion(self):
        # Pangolin rejected CTG>CT outright rather than scoring it, so this is what the old formula would
        # have done, not what it did: zero the T, a base CTG>CT does not touch. It is why trimming has to
        # come first now that these spellings are accepted.
        _, alt_scores, _, aligned_alt = self.align("CTG", "CT")
        self.assertFalse(np.array_equal(aligned_alt, legacy_alt_scores(alt_scores, "CTG", "CT", D)))
        self.assertEqual(legacy_alt_scores(alt_scores, "CTG", "CT", D)[D+1], 0)
        self.assertNotEqual(aligned_alt[D+1], 0)

    def test_padded_insertion_is_realigned_where_the_alleles_differ(self):
        # CA>CAT is A>AT one base later: the C keeps its own score
        ref_scores, alt_scores, aligned_ref, aligned_alt = self.align("CA", "CAT")
        np.testing.assert_array_equal(aligned_ref, ref_scores)
        np.testing.assert_array_equal(aligned_alt, np.concatenate(
            [alt_scores[:D+1], np.max(alt_scores[D+1:D+3], keepdims=True), alt_scores[D+3:]]))

    # --- a deletion-insertion reports its whole span once, at the span's first position ---

    def test_deletion_insertion_reports_the_strongest_site_on_each_side(self):
        ref_scores, alt_scores, aligned_ref, aligned_alt = self.align("AT", "GCC")
        self.assertEqual(aligned_ref[D], np.max(ref_scores[D:D+2]))
        self.assertEqual(aligned_alt[D], np.max(alt_scores[D:D+3]))

    def test_deletion_insertion_leaves_the_rest_of_the_span_showing_no_change(self):
        ref_scores, _, aligned_ref, aligned_alt = self.align("ATG", "GC")
        np.testing.assert_array_equal(aligned_ref[D+1:D+3], ref_scores[D+1:D+3])
        np.testing.assert_array_equal(aligned_alt[D+1:D+3], ref_scores[D+1:D+3])

    def test_deletion_insertion_leaves_positions_outside_the_span_alone(self):
        ref_scores, alt_scores, aligned_ref, aligned_alt = self.align("AT", "GCC")
        np.testing.assert_array_equal(aligned_ref[:D], ref_scores[:D])
        np.testing.assert_array_equal(aligned_ref[D+2:], ref_scores[D+2:])
        np.testing.assert_array_equal(aligned_alt[:D], alt_scores[:D])
        np.testing.assert_array_equal(aligned_alt[D+2:], alt_scores[D+3:])

    def test_deletion_insertion_with_shared_bases_is_reported_at_the_bases_it_changes(self):
        # GAT>GGCC changes AT>GCC one base later, so the span starts one base after the center
        ref_scores, alt_scores, aligned_ref, aligned_alt = self.align("GAT", "GGCC")
        np.testing.assert_array_equal(aligned_ref[:D+1], ref_scores[:D+1])
        self.assertEqual(aligned_ref[D+1], np.max(ref_scores[D+1:D+3]))
        self.assertEqual(aligned_alt[D+1], np.max(alt_scores[D+1:D+4]))

    def test_output_has_one_score_per_ref_position(self):
        # including a REF far longer than the window is wide: Pangolin scores d bases past the end of
        # the REF allele, so the bases a variant changes are always inside the window
        for ref, alt in (("G", "A"), ("TG", "TA"), ("GGGC", "G"), ("A", "AGAGAG"), ("CTG", "CT"),
                         ("CA", "CAT"), ("AT", "GCC"), ("ATG", "GC"), ("GAT", "GGCC"),
                         ("A"*600, "GC"), ("A"*600, "A"), ("A"*600, "A"*597 + "GC")):
            with self.subTest(ref=ref, alt=alt):
                _, _, aligned_ref, aligned_alt = self.align(ref, alt)
                self.assertEqual(len(aligned_ref), 2*D + len(ref))
                self.assertEqual(len(aligned_alt), 2*D + len(ref))


class RefAndAltBasesForPositionTest(unittest.TestCase):

    POS = 100

    def bases(self, ref, alt, genomic_coord, reference_base):
        return ref_and_alt_bases_for_position(self.POS, ref, alt, genomic_coord, reference_base)

    def test_snv_shows_the_alt_base(self):
        self.assertEqual(self.bases("G", "A", 100, "G"), ("G", "A"))

    def test_mnv_shows_one_alt_base_per_position(self):
        self.assertEqual(self.bases("TG", "CA", 100, "T"), ("T", "C"))
        self.assertEqual(self.bases("TG", "CA", 101, "G"), ("G", "A"))

    def test_deletion_shows_the_alleles_on_the_anchor_and_dashes_the_deleted_bases(self):
        self.assertEqual(self.bases("TG", "T", 100, "T"), ("TG", "T"))
        self.assertEqual(self.bases("TG", "T", 101, "G"), ("G", "-"))

    def test_insertion_shows_the_alleles_on_the_anchor(self):
        self.assertEqual(self.bases("A", "AGA", 100, "A"), ("A", "AGA"))

    def test_padded_snv_is_reported_where_the_alleles_differ(self):
        # TG>TA changes only the second base, so the first shows no change
        self.assertEqual(self.bases("TG", "TA", 100, "T"), ("T", "T"))
        self.assertEqual(self.bases("TG", "TA", 101, "G"), ("G", "A"))

    def test_padded_deletion_is_reported_where_the_alleles_differ(self):
        # CTG>CT is TG>T one base later
        self.assertEqual(self.bases("CTG", "CT", 100, "C"), ("C", "C"))
        self.assertEqual(self.bases("CTG", "CT", 101, "T"), ("TG", "T"))
        self.assertEqual(self.bases("CTG", "CT", 102, "G"), ("G", "-"))

    def test_deletion_insertion_shows_the_alleles_on_the_anchor(self):
        self.assertEqual(self.bases("AT", "GCC", 100, "A"), ("AT", "GCC"))
        self.assertEqual(self.bases("AT", "GCC", 101, "T"), ("T", "-"))

    def test_positions_outside_the_variant_show_the_reference_base_on_both_sides(self):
        self.assertEqual(self.bases("TG", "T", 99, "C"), ("C", "C"))
        self.assertEqual(self.bases("TG", "T", 102, "A"), ("A", "A"))


if __name__ == "__main__":
    unittest.main()
