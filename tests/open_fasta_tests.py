"""Tests for open_fasta in pangolin/pangolin.py.

Importing pangolin.pangolin needs its full dependencies (torch, pyfastx, gffutils, pandas, PyVCF), but not the models.

Run from the repo root with:  python3 -m unittest tests.open_fasta_tests -v
"""

import multiprocessing
import os
import tempfile
import threading
import unittest

from pangolin.pangolin import open_fasta

FASTA_CONTENTS = ">chr1\nACGTACGTAC\n>chr2\nGGGGCCCCAA\n"


def write_fasta(directory, filename):
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(FASTA_CONTENTS)
    return path


def report_whether_child_opened_its_own_handle(reference_file, parent_handle, queue):
    """Runs in a forked child, which inherits the parent's handle cache along with its memory."""
    child_handle = open_fasta(reference_file)
    queue.put((child_handle is not parent_handle, child_handle["chr2"][0:4].seq))


class OpenFastaTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reference_file = write_fasta(self.temp_dir.name, "reference.fa")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_a_handle_reads_the_right_sequence(self):
        self.assertEqual(open_fasta(self.reference_file)["chr1"][2:6].seq, "GTAC")
        self.assertEqual(open_fasta(self.reference_file)["chr2"][4:10].seq, "CCCCAA")

    def test_the_same_thread_gets_the_same_handle_back(self):
        self.assertIs(open_fasta(self.reference_file), open_fasta(self.reference_file))

    def test_each_reference_file_gets_its_own_handle(self):
        other_reference_file = write_fasta(self.temp_dir.name, "other_reference.fa")
        self.assertIsNot(open_fasta(self.reference_file), open_fasta(other_reference_file))

    def test_each_thread_gets_its_own_handle(self):
        main_thread_handle = open_fasta(self.reference_file)
        handles_and_sequences_by_thread = []

        def open_in_thread():
            handle = open_fasta(self.reference_file)
            handles_and_sequences_by_thread.append((handle, open_fasta(self.reference_file) is handle, handle["chr1"][0:4].seq))

        threads = [threading.Thread(target=open_in_thread) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        handles = [main_thread_handle] + [handle for handle, _, _ in handles_and_sequences_by_thread]
        self.assertEqual(len({id(handle) for handle in handles}), 5)
        for _, reused_within_thread, sequence in handles_and_sequences_by_thread:
            self.assertTrue(reused_within_thread)
            self.assertEqual(sequence, "ACGT")

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "needs fork()")
    def test_a_forked_child_opens_its_own_handle(self):
        parent_handle = open_fasta(self.reference_file)
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        child = context.Process(
            target=report_whether_child_opened_its_own_handle, args=(self.reference_file, parent_handle, queue))
        child.start()
        child_opened_its_own_handle, child_sequence = queue.get(timeout=60)
        child.join()

        self.assertEqual(child.exitcode, 0)
        self.assertTrue(child_opened_its_own_handle)
        self.assertEqual(child_sequence, "GGGG")
        self.assertIs(open_fasta(self.reference_file), parent_handle)


if __name__ == "__main__":
    unittest.main()
