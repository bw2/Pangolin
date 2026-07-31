import argparse
from pkg_resources import resource_filename
import gffutils
import numpy as np
import pandas as pd
import pyfastx
import torch
import vcf

from pangolin.model import L, W, AR, Pangolin

FLOAT_FORMAT = "0.2f"

# use a 0.1 threshold for ALL_NON_ZERO_SCORES because Pangolin's baseline probability seems to be ~0.05 for most
# positions, so 0.1 separates the unusually large scores
MIN_SCORE_THRESHOLD = 0.1

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def one_hot_encode(seq, strand):
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]


def compute_score(ref_seq, alt_seq, strand, d, models):
    ref_seq = one_hot_encode(ref_seq, strand).T
    ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()
    alt_seq = one_hot_encode(alt_seq, strand).T
    alt_seq = torch.from_numpy(np.expand_dims(alt_seq, axis=0)).float()

    if torch.cuda.is_available():
        ref_seq = ref_seq.to(torch.device("cuda"))
        alt_seq = alt_seq.to(torch.device("cuda"))

    pangolin = []
    pangolin_ref = []
    pangolin_alt = []
    for j in range(4):
        score = []
        score_ref = []
        score_alt = []
        for model in models[3*j: 3*j+3]:
            with torch.no_grad():
                ref = model(ref_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                alt = model(alt_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                if strand == '-':
                    ref = ref[::-1]
                    alt = alt[::-1]

                l = 2*d+1
                ndiff = np.abs(len(ref)-len(alt))
                if len(ref) > len(alt):
                    alt = np.concatenate([alt[0:l//2+1], np.zeros(ndiff), alt[l//2+1:]])
                elif len(ref) < len(alt):
                    alt = np.concatenate([alt[0:l//2], np.max(alt[l//2:l//2+ndiff+1], keepdims=True), alt[l//2+ndiff+1:]])

                score.append(alt - ref)
                score_ref.append(ref)
                score_alt.append(alt)

        pangolin.append(np.mean(score, axis=0))
        pangolin_ref.append(np.mean(score_ref, axis=0))
        pangolin_alt.append(np.mean(score_alt, axis=0))

    pangolin = np.array(pangolin)
    pangolin_ref = np.array(pangolin_ref)
    pangolin_alt = np.array(pangolin_alt)
    pangolin_argmin = np.argmin(pangolin, axis=0)
    pangolin_argmax = np.argmax(pangolin, axis=0)
    pangolin_idx = np.arange(pangolin.shape[1])

    # Since pangolin computes max gain scores across 4 tissues and, separately, max loss scores across the 4 tissues,
    # the ref and alt probabilities for the gain scores can be from a different tissue than the splice loss probablities.
    # Therefore, we need to keep the ref and alt probabilities that underlie the gain score, and, separately, also the
    # ref and alt probabilities that underlie the loss score.
    loss = pangolin[pangolin_argmin, pangolin_idx]   # this is a 1d array that should == the difference between loss_ref and loss_alt
    loss_ref = pangolin_ref[pangolin_argmin, pangolin_idx]   # at each position, select the ref sequence splice probability from the tissue that had the maximum loss score
    loss_alt = pangolin_alt[pangolin_argmin, pangolin_idx]   # at each position, select the alt sequence splice probability from the tissue that had the maximum loss score

    # basic internal consistency checks
    if len(loss) != len(loss_alt) or len(loss) != len(loss_ref):
        raise ValueError(f"len(loss) != len(loss_alt) or len(loss) != len(loss_ref): {len(loss)} != {len(loss_alt)} or {len(loss)} != {len(loss_ref)}")
    if any(abs(l - (a - r)) > 1e-5 for l, a, r in zip(loss, loss_alt, loss_ref)):
        raise ValueError("Internal error: loss != loss_alt - loss_ref")

    gain = pangolin[pangolin_argmax, pangolin_idx]   # this is 1d array that should == the difference between gain_ref and gain_alt
    gain_ref = pangolin_ref[pangolin_argmax, pangolin_idx]
    gain_alt = pangolin_alt[pangolin_argmax, pangolin_idx]
    # basic internal consistency checks
    if len(gain) != len(gain_alt) or len(gain) != len(gain_ref):
        raise ValueError(f"len(gain) != len(gain_alt) or len(gain) != len(gain_ref): {len(gain)} != {len(gain_alt)} or {len(gain)} != {len(gain_ref)}")
    if any(abs(g - (a - r)) > 1e-5 for g, a, r in zip(gain, gain_alt, gain_ref)):
        raise ValueError("Internal error: gain != gain_alt - gain_ref")

    return loss, gain, loss_ref, loss_alt, gain_ref, gain_alt


def compute_ref_score(ref_seq, strand, models):
    """REF-only counterpart of compute_score: no ALT sequence needed.

    Returns, per position, the strongest predicted splice-site probability
    across the 4 tissue models (Pangolin does not distinguish acceptor vs.
    donor the way SpliceAI does -- each tissue model just predicts "is this a
    splice site").
    """
    ref_seq = one_hot_encode(ref_seq, strand).T
    ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()

    if torch.cuda.is_available():
        ref_seq = ref_seq.to(torch.device("cuda"))

    pangolin_ref = []
    for j in range(4):
        score_ref = []
        for model in models[3*j: 3*j+3]:
            with torch.no_grad():
                ref = model(ref_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                if strand == '-':
                    ref = ref[::-1]
                score_ref.append(ref)
        pangolin_ref.append(np.mean(score_ref, axis=0))

    pangolin_ref = np.array(pangolin_ref)
    return np.max(pangolin_ref, axis=0)


def process_position(lnum, chr, pos, gtf, models, args):
    """REF-only counterpart of process_variant: no ALT allele needed.

    Reports, per transcript, the strongest predicted REF splice-site
    probability anywhere in the +/-args.distance window around `pos` (matching
    the "max score in window" convention used by SpliceAI's REF-only mode),
    plus the full above-threshold curve for visualization.
    """
    d = args.distance

    fasta = pyfastx.Fasta(args.reference_file)
    if chr not in fasta.keys() and "chr"+chr in fasta.keys():
        chr = "chr"+chr
    elif chr not in fasta.keys() and chr[3:] in fasta.keys():
        chr = chr[3:]

    try:
        seq = fasta[chr][pos-5001-d:pos+5000+d].seq
    except Exception as e:
        print(e)
        print("[Line %s]" % lnum, "WARNING, skipping position: Could not get sequence, possibly because the position is too close to chromosome ends. "
                                  "See error message above.")
        return None

    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos) + len(genes_neg) == 0:
        print("[Line %s]" % lnum, "WARNING, skipping position: Not contained in a gene body. Do GTF/FASTA chromosome names match?")
        return None

    genomic_coords = np.arange(pos-d, pos+d+1)

    results = []
    for genes, strand in [(genes_pos, "+"), (genes_neg, "-")]:
        if not genes:
            continue

        ref_score = compute_ref_score(seq, strand, models)

        if len(genomic_coords) != len(ref_score):
            raise ValueError(f"Internal error: len(genomic_coords) != len(ref_score): {len(genomic_coords)} != {len(ref_score)}")

        s = np.argmax(ref_score)
        for transcript_id in genes:
            results.append({
                "NAME": transcript_id,
                "S_REF": f"{ref_score[s]:{FLOAT_FORMAT}}",  # strongest predicted REF splice-site probability anywhere in the window
                "DP_S": int(s-d),  # position (relative to the queried position) of that strongest predicted site
                "ALL_NON_ZERO_SCORES": [
                    {"pos": int(genomic_coord), "S_REF": f"{score:{FLOAT_FORMAT}}"}
                    for i, (genomic_coord, score) in enumerate(zip(genomic_coords, ref_score))
                    if score >= MIN_SCORE_THRESHOLD or i == s
                ],
                "STRAND": strand,
            })

    return results


def get_genes(chr, pos, gtf):
    genes = gtf.region((chr, pos-1, pos-1), featuretype="transcript")
    genes_pos, genes_neg = {}, {}

    for gene in genes:
        if gene[3] > pos or gene[4] < pos:
            continue
        transcript_id = gene["transcript_id"][0]
        exons = []
        for exon in gtf.children(gene, featuretype="exon"):
            exons.extend([exon[3], exon[4]])
        if gene[6] == '+':
            genes_pos[transcript_id] = exons
        elif gene[6] == '-':
            genes_neg[transcript_id] = exons

    return genes_pos, genes_neg


def process_variant(lnum, chr, pos, ref, alt, gtf, models, args):
    d = args.distance

    if len(set("ACGT").intersection(set(ref))) == 0 or len(set("ACGT").intersection(set(alt))) == 0 \
            or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt)):
        print("[Line %s]" % lnum, "WARNING, skipping variant: Variant format not supported.")
        return None
    elif len(ref) > 2*d:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Deletion too large")
        return None

    fasta = pyfastx.Fasta(args.reference_file)
    # try to make vcf chromosomes compatible with reference chromosomes
    if chr not in fasta.keys() and "chr"+chr in fasta.keys():
        chr = "chr"+chr
    elif chr not in fasta.keys() and chr[3:] in fasta.keys():
        chr = chr[3:]

    try:
        seq = fasta[chr][pos-5001-d:pos+len(ref)+4999+d].seq
    except Exception as e:
        print(e)
        print("[Line %s]" % lnum, "WARNING, skipping variant: Could not get sequence, possibly because the variant is too close to chromosome ends. "
                                  "See error message above.")
        return None

    if seq[5000+d:5000+d+len(ref)] != ref:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Mismatch between FASTA (ref base: %s) and variant file (ref base: %s)."
              % (seq[5000+d:5000+d+len(ref)], ref))
        return None

    ref_seq = seq
    alt_seq = seq[:5000+d] + alt + seq[5000+d+len(ref):]

    # get genes that intersect variant
    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos) + len(genes_neg) == 0:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Variant not contained in a gene body. Do GTF/FASTA chromosome names match?")
        return None

    # get splice scores
    genomic_coords = np.arange(pos-d, pos+d+len(ref))

    results = []
    for genes, strand in [(genes_pos, "+"), (genes_neg, "-")]:
        if not genes:
            continue

        orig_loss, orig_gain, loss_ref, loss_alt, gain_ref, gain_alt = compute_score(ref_seq, alt_seq, strand, d, models)

        for transcript_id, positions in genes.items():
            positions = np.array(positions)
            positions = positions - (pos - d)

            if args.mask != "True":
                loss = orig_loss
                gain = orig_gain
            else:
                # Make copies of the loss/gain for each gene to avoid overwriting data between genes
                loss = np.copy(orig_loss)
                gain = np.copy(orig_gain)

                if len(positions) != 0:
                    positions_filt = positions[(positions >= 0) & (positions < len(loss))]
                    # set splice gain at annotated sites to 0
                    gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                    # set splice loss at unannotated sites to 0
                    not_positions = ~np.isin(np.arange(len(loss)), positions_filt)
                    loss[not_positions] = np.maximum(loss[not_positions], 0)

                else:
                    loss[:] = np.maximum(loss[:], 0)

            if len(genomic_coords) != len(gain):
                raise ValueError(f"Internal error: len(genomic_coords) != len(gain): {len(genomic_coords)} != {len(gain)}")
            if len(genomic_coords) != len(loss):
                raise ValueError(f"Internal error: len(genomic_coords) != len(loss): {len(genomic_coords)} != {len(loss)}")

            l, g = np.argmin(loss), np.argmax(gain)
            results.append({
                "NAME": transcript_id,
                "DS_SG": f"{gain[g]:{FLOAT_FORMAT}}",  # splice gain delta score at the position where the splice gain delta score is maximum
                "DS_SL": f"{loss[l]:{FLOAT_FORMAT}}",  # splice loss delta score at the position where the splice loss delta score is maximum
                "DP_SG": int(g-d),   # relative position where the splice gain delta score is maximum
                "DP_SL": int(l-d),   # relative position where the splice loss delta score is maximum
                "SG_REF": f"{gain_ref[g]:{FLOAT_FORMAT}}",  # reference sequence splice probability at position and tissue where splice gain is maximum
                "SG_ALT": f"{gain_alt[g]:{FLOAT_FORMAT}}",  # alt sequence splice probability at position and tissue where splice gain is maximum
                "SL_REF": f"{loss_ref[l]:{FLOAT_FORMAT}}",  # reference sequence splice probability at position and tissue where splice loss is maximum
                "SL_ALT": f"{loss_alt[l]:{FLOAT_FORMAT}}",  # alt sequence splice probability at position and tissue where splice loss is maximum
                "ALL_NON_ZERO_SCORES": [
                    {
                        "pos": int(genomic_coord),
                        "SL_REF": f"{loss_ref_score:{FLOAT_FORMAT}}",  # reference sequence splice probability in the tissue where the splice loss delta score is largest at this position
                        "SL_ALT": f"{loss_alt_score:{FLOAT_FORMAT}}",  # alt sequence splice probability in the tissue where the splice loss delta score is largest at this position
                        "SG_REF": f"{gain_ref_score:{FLOAT_FORMAT}}",  # reference sequence splice probability in the tissue where the splice gain delta score is largest at this position
                        "SG_ALT": f"{gain_alt_score:{FLOAT_FORMAT}}",  # alt sequence splice probability in the tissue where the splice gain delta score is largest at this position
                    } for i, (genomic_coord, loss_ref_score, loss_alt_score, gain_ref_score, gain_alt_score) in enumerate(zip(
                        genomic_coords, loss_ref, loss_alt, gain_ref, gain_alt)
                    ) if any(score >= MIN_SCORE_THRESHOLD for score in (
                        loss_ref_score, loss_alt_score, gain_ref_score, gain_alt_score)) or i in (l, g)
                ],
                "STRAND": strand,
            })

    return results


def convert_scores_to_string(scores):
    return ",".join([
        "|".join([s["NAME"], f"{s['DP_SG']}:{s['DS_SG']}", f"{s['DP_SL']}:{s['DS_SL']}"]) for s in scores
    ])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("variant_file", help="VCF or CSV file with a header (see COLUMN_IDS option).")
    parser.add_argument("reference_file", help="FASTA file containing a reference genome sequence.")
    parser.add_argument("annotation_file", help="gffutils database file. Can be generated using create_db.py.")
    parser.add_argument("output_file", help="Prefix for output file. Will be a VCF/CSV if variant_file is VCF/CSV.")
    parser.add_argument("-c", "--column_ids", default="CHROM,POS,REF,ALT", help="(If variant_file is a CSV) Column IDs for: chromosome, variant position, reference bases, and alternative bases. "
                                                                                "Separate IDs by commas. (Default: CHROM,POS,REF,ALT)")
    parser.add_argument("-m", "--mask", default="True", choices=["False","True"], help="If True, splice gains (increases in score) at annotated splice sites and splice losses (decreases in score) at unannotated splice sites will be set to 0. (Default: True)")
    parser.add_argument("-s", "--score_cutoff", type=float, help="Output all sites with absolute predicted change in score >= cutoff, instead of only the maximum loss/gain sites.")
    parser.add_argument("-d", "--distance", type=int, default=50, help="Number of bases on either side of the variant for which splice scores should be calculated. (Default: 50)")
    parser.add_argument("--score_exons", default="False", choices=["False","True"], help="Output changes in score for both splice sites of annotated exons, as long as one splice site is within the considered range (specified by -d). Output will be: gene|site1_pos:score|site2_pos:score|...")
    args = parser.parse_args()

    variants = args.variant_file
    gtf = args.annotation_file
    try:
        gtf = gffutils.FeatureDB(gtf)
    except:
        print("ERROR, annotation_file could not be opened. Is it a gffutils database file?")
        exit()

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    models = []
    for i in [0, 2, 4, 6]:
        for j in range(1, 4):
            model = Pangolin(L, W, AR)
            if torch.cuda.is_available():
                model.cuda()
                weights = torch.load(resource_filename(__name__,"models/final.%s.%s.3.v2" % (j, i)))
            else:
                weights = torch.load(resource_filename(__name__,"models/final.%s.%s.3.v2" % (j, i)), map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)

    if variants.endswith(".vcf"):
        lnum = 0
        # count the number of header lines
        for line in open(variants, 'r'):
            lnum += 1
            if line[0] != '#':
                break

        variants = vcf.Reader(filename=variants)
        variants.infos["Pangolin"] = vcf.parser._Info(
            "Pangolin",'.',"String","Pangolin splice scores. "
            "Format: gene|pos:score_change|pos:score_change|...",'.','.')
        fout = vcf.Writer(open(args.output_file+".vcf", 'w'), variants)

        for i, variant in enumerate(variants):
            scores = process_variant(lnum+i, str(variant.CHROM), int(variant.POS), variant.REF, str(variant.ALT[0]), gtf, models, args)
            if scores:
                variant.INFO["Pangolin"] = convert_scores_to_string(scores)
            fout.write_record(variant)
            fout.flush()

        fout.close()

    elif variants.endswith(".csv"):
        col_ids = args.column_ids.split(',')
        variants = pd.read_csv(variants, header=0)
        fout = open(args.output_file+".csv", 'w')
        fout.write(','.join(variants.columns)+',Pangolin\n')
        fout.flush()

        for lnum, variant in variants.iterrows():
            chr, pos, ref, alt = variant[col_ids]
            ref, alt = ref.upper(), alt.upper()
            scores = process_variant(lnum+1, str(chr), int(pos), ref, alt, gtf, models, args)

            if not scores:
                fout.write(','.join(variant.to_csv(header=False, index=False).split('\n'))+'\n')
            else:
                fout.write(','.join(variant.to_csv(header=False, index=False).split('\n'))+convert_scores_to_string(scores)+'\n')
            fout.flush()

        fout.close()

    else:
        print("ERROR, variant_file needs to be a CSV or VCF.")


if __name__ == '__main__':
    main()
