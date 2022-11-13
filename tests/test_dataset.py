from unittest import TestCase
import unittest 
from hydra import compose, initialize
from omegaconf import OmegaConf
from datasets import ClassificationDataset

####TODO###
class TestDataset(TestCase):

    def setUp(self):
        cfg = OmegaConf.load("cfg/config.yaml")
        self.path = cfg.dataset.path


    def test_data_labels(self):
        dataloader = ClassificationDataset(self.path)
        df_data = dataloader._create_df()
        df_data.label.unique()
        