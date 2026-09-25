"""Line up Pangolin's REF and ALT scores so that they can be compared position by position.

Kept out of pangolin.py, which loads torch and the 12 models on import, so that it can be tested
with numpy alone:
    python3 -m unittest tests.score_alignment_tests -v

The same rules are implemented for SpliceAI in spliceai/score_alignment.py of the fork at
https://github.com/bw2/SpliceAI, so that both tools report a given variant the same way.

Pangolin scores the positions pos-d through pos+d+len(ref)-1, which is 2*d + len(ref) of them,
with the variant's first base at index d. Because that window extends d bases past the end of the
REF allele, the bases a variant changes always lie inside it, so there is no counterpart here to
SpliceAI's span_fits_in_output_window: SpliceAI scores a fixed 2*d + 1 positions centred on the
variant, which a long REF allele can reach past.
"""

import numpy as np


def trim_shared_bases(ref, alt):
    """Trim the bases REF and ALT share, leaving the bases the variant actually changes.

    Bases shared at the end are dropped first, then bases shared at the start, and each allele always
    keeps at least one base, so an insertion or deletion keeps its anchor base the way VCF writes it.

    Args:
        ref (str): REF allele
        alt (str): ALT allele

    Returns:
        tuple: (number of bases dropped from the start, trimmed REF, trimmed ALT)
    """
    bases_dropped_from_start = 0
    while len(ref) > 1 and len(alt) > 1 and ref[-1].upper() == alt[-1].upper():
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0].upper() == alt[0].upper():
        ref, alt = ref[1:], alt[1:]
        bases_dropped_from_start += 1
    return bases_dropped_from_start, ref, alt


def get_padded_sequence(contig, start, end):
    """Read the bases from `start` to `end` of a contig, filling in N for any that lie past either end.

    The model reads a fixed 5,000 bases of context on each side of the window it scores, and near a
    contig end the FASTA has fewer than that. Records there used to be skipped. N is what the model
    is given for an unknown base (one_hot_encode maps it to all zeros, as SpliceAI does for every
    base outside the annotated gene), so the bases the contig doesn't have are filled in the same way
    and the sequence always comes back at the length the caller asked for. The start index is clamped
    rather than passed through because pyfastx segfaults on a negative one; past the end it simply
    returns fewer bases.

    Args:
        contig: a pyfastx sequence (fasta[chr])
        start (int): 0-based start index, which may be negative
        end (int): 0-based exclusive end index, which may lie past the end of the contig

    Returns:
        str: exactly end - start bases
    """
    clamped_start = max(start, 0)
    seq = contig[clamped_start:end].seq
    return 'N' * (clamped_start - start) + seq + 'N' * (end - clamped_start - len(seq))


def align_ref_and_alt_scores(ref_scores, alt_scores, ref, alt, d, anchor_scores=None):
    """Line up the model's scores for the ALT sequence with the REF positions they are compared against.

    When REF and ALT differ in length, the ALT sequence has more or fewer positions than the REF one, so
    its scores are collapsed onto the REF positions before the two are subtracted. The bases the alleles
    share are trimmed off first, so which positions the variant changes does not depend on how many unchanged
    bases it was written with. The window itself is still anchored on the position as written, so a padded
    spelling covers a few extra positions at its edges, and two spellings agree at every position they share.
    A spelling padded with shared bases, such as CTG>CT rather than TG>T, used to be rejected
    as "Variant format not supported" rather than mis-scored, so every variant Pangolin scored before this
    change still scores exactly the same.

    After trimming:
    - same length (an SNV or MNV): the positions already line up
    - one-base ALT (a deletion): the anchor base keeps its score and the deleted bases get zero
    - one-base REF (an insertion): the anchor base gets the highest score among itself and the inserted bases
    - otherwise (a deletion-insertion): the whole span is reported once, at the position holding the
      highest REF score in it, where the ALT track carries the highest score among the bases put in the
      span's place, so the difference between the two is what the variant changed. Every other position
      of the span is given its REF score on both tracks, which reports no change there. Reporting at the
      strongest REF position rather than at the span's first base keeps a splice site the variant
      replaces at its own coordinate, which is what masking needs, since it keeps a loss only where a
      splice site is annotated.

    The first three cases reproduce Pangolin's original handling exactly for a variant already written
    with no shared bases. The fourth is new: those variants used to be rejected as "Variant format not
    supported".

    Args:
        ref_scores (numpy.ndarray): REF scores in genomic order, of length 2*d + len(ref)
        alt_scores (numpy.ndarray): ALT scores in genomic order, of length 2*d + len(alt), with the
            variant's first base at index d
        ref (str): REF allele
        alt (str): ALT allele
        d (int): number of bases on either side of the variant that are being scored
        anchor_scores (numpy.ndarray): REF scores to pick the deletion-insertion's position from, for
            callers that have several score tracks to line up and then average together. They have to
            report the span at one position or the average spreads a single signal over several of them
            and shrinks it. Defaults to ref_scores, which is right for a caller with one track.

    Returns:
        tuple: (ref_scores, alt_scores), each of length 2*d + len(ref), one score per REF position
    """
    bases_dropped_from_start, ref, alt = trim_shared_bases(ref, alt)
    start = d + bases_dropped_from_start
    if len(ref) == len(alt):
        return ref_scores, alt_scores

    if len(alt) == 1:
        return ref_scores, np.concatenate([
            alt_scores[:start+1],
            np.zeros(len(ref)-1),
            alt_scores[start+1:]])

    if len(ref) == 1:
        return ref_scores, np.concatenate([
            alt_scores[:start],
            np.max(alt_scores[start:start+len(alt)], keepdims=True),
            alt_scores[start+len(alt):]])

    # A deletion-insertion replaces every base of the span at once, so no base inside it has a counterpart
    # to be compared against. The comparison for the whole span is reported at one position: the strongest
    # REF base of anchor_scores, which is this track unless the caller supplies another. That keeps a
    # splice site the variant replaces at its own coordinate; np.argmax takes the first maximum, which is
    # the earliest genomic position when several tie. Every other position of the span holds its REF score
    # on both tracks, so the difference between the tracks, which is what the caller reports as a change,
    # is zero there. The REF track is returned unchanged, since every position already carries its own
    # score and only the ALT side has to move.
    span_ref = ref_scores[start:start+len(ref)]
    anchor_span = span_ref if anchor_scores is None else anchor_scores[start:start+len(ref)]
    alt_span = span_ref.copy()
    alt_span[np.argmax(anchor_span)] = np.max(alt_scores[start:start+len(alt)])
    return ref_scores, np.concatenate([alt_scores[:start], alt_span, alt_scores[start+len(alt):]])


def ref_and_alt_bases_for_position(pos, ref, alt, genomic_coord, reference_base):
    """Pick the REF and ALT bases to show on one row of the per-position table.

    The alleles are named where they first differ, which is not the variant's own position when the
    variant is written with shared leading bases, so the alleles shown there are the trimmed ones. For
    a variant already written with no shared bases they are the whole alleles.

    That is also where the scores sit, except for a deletion-insertion: its comparison is reported at
    the strongest REF base of the span (see align_ref_and_alt_scores), which can be a later one. Every
    base of the span is labelled as replaced, so the row carrying the comparison can read "<base> / -"
    while the row naming the alleles shows no change. A plain deletion already reports that way.

    Args:
        pos (int): 1-based position of the variant
        ref (str): REF allele
        alt (str): ALT allele
        genomic_coord (int): 1-based position of the row
        reference_base (str): the reference genome's base at genomic_coord

    Returns:
        tuple: (REF base to show, ALT base to show)
    """
    bases_dropped_from_start, trimmed_ref, trimmed_alt = trim_shared_bases(ref, alt)
    changed_span_start = pos + bases_dropped_from_start

    if genomic_coord == changed_span_start and len(trimmed_ref) != len(trimmed_alt):
        # insertion or deletion: name the alleles where they first differ
        return trimmed_ref, trimmed_alt

    if changed_span_start <= genomic_coord < changed_span_start + len(trimmed_ref):
        # one of the bases the variant replaces: for a change of the same length each position
        # has its own ALT base, otherwise the base is deleted by the variant
        if len(trimmed_ref) == len(trimmed_alt):
            return reference_base, trimmed_alt[genomic_coord - changed_span_start]
        return reference_base, "-"

    return reference_base, reference_base
