"""Notebook cancellation reaches separately spawned model groups, not bystanders."""
import unittest
from unittest.mock import patch
from tools.process_groups import descendant_groups


class OwnedProcessGroupsTests(unittest.TestCase):
    def test_nested_model_sessions_and_workers_are_owned_but_kernel_is_not(self):
        table='10 1 10\n20 10 20\n21 20 20\n30 20 30\n31 30 30\n40 1 40\n'
        with patch('tools.process_groups.subprocess.check_output',return_value=table),patch('tools.process_groups.os.getpgrp',return_value=10):
            self.assertEqual(descendant_groups(20),{20,30})
        with patch('tools.process_groups.subprocess.check_output',return_value=table),patch('tools.process_groups.os.getpgrp',return_value=10):
            self.assertEqual(descendant_groups(10),{20,30})


if __name__=='__main__':unittest.main()
