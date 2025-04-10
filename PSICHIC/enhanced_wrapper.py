# -*- coding: utf-8 -*-
import json
import os
import pandas as pd
import torch

from psichic_utils.dataset import ProteinMoleculeDataset
from psichic_utils.data_utils import DataLoader, virtual_screening
from psichic_utils import protein_init, ligand_init
from models.net import net
from runtime_config import RuntimeConfig

class EnhancedPsichicWrapper:
    def __init__(self):
        """Load the model config and do a one-time model instantiation."""
        self.runtime_config = RuntimeConfig()
        self.device = self.runtime_config.DEVICE

        # Load model config
        config_path = os.path.join(self.runtime_config.MODEL_PATH, 'config.json')
        with open(config_path, 'r') as f:
            self.model_config = json.load(f)

        self.model = None
        self.protein_dict = None
        self.protein_seq = []
        self.smiles_dict = None
        self.smiles_list = []

        # Actually load model
        self._load_model_once()

    def _load_model_once(self):
        """Loads the model from disk and sets eval mode, so it's done only once."""
        degree_dict_path = os.path.join(self.runtime_config.MODEL_PATH, 'degree.pt')
        model_weights_path = os.path.join(self.runtime_config.MODEL_PATH, 'model.pt')

        # Load degree info
        degree_dict = torch.load(degree_dict_path, map_location=self.device)
        mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']

        # Instantiate net
        self.model = net(
            mol_deg, prot_deg,
            mol_in_channels=self.model_config['params']['mol_in_channels'],
            prot_in_channels=self.model_config['params']['prot_in_channels'],
            prot_evo_channels=self.model_config['params']['prot_evo_channels'],
            hidden_channels=self.model_config['params']['hidden_channels'],
            pre_layers=self.model_config['params']['pre_layers'],
            post_layers=self.model_config['params']['post_layers'],
            aggregators=self.model_config['params']['aggregators'],
            scalers=self.model_config['params']['scalers'],
            total_layer=self.model_config['params']['total_layer'],
            K=self.model_config['params']['K'],
            heads=self.model_config['params']['heads'],
            dropout=self.model_config['params']['dropout'],
            dropout_attn_score=self.model_config['params']['dropout_attn_score'],
            regression_head=self.model_config['tasks']['regression_task'],
            classification_head=self.model_config['tasks']['classification_task'],
            multiclassification_head=self.model_config['tasks']['mclassification_task'],
            device=self.device
        ).to(self.device)

        self.model.reset_parameters()

        # Load weights
        state = torch.load(model_weights_path, map_location=self.device)
        self.model.load_state_dict(state)

        # Set to eval mode
        self.model.eval()

        # Optionally use half precision for speed (if stable):
        # if 'cuda' in str(self.device).lower():
        #     self.model.half()

    def initialize_protein(self, protein_seq: str) -> dict:
        """Parse and store the new protein sequence without reloading the model."""
        self.protein_seq = [protein_seq]
        protein_dict = protein_init(self.protein_seq)
        return protein_dict

    def initialize_proteins_batch(self, protein_sequences: list) -> dict:
        """
        NEW METHOD: Initialize multiple proteins at once.
        This is more efficient than processing one at a time.
        """
        self.protein_seq = protein_sequences
        self.protein_dict = protein_init(protein_sequences)
        return self.protein_dict

    def initialize_smiles(self, smiles_list: list) -> dict:
        """
        Pre-featurize a list of SMILES strings just once.
        This can be done globally so we skip re-initializing them each time.
        """
        self.smiles_list = smiles_list
        smiles_dict = ligand_init(smiles_list)
        return smiles_dict

    def run_challenge_start(self, protein_seq: str):
        """
        Called before validation to set the current protein dict.
        No repeated model loading from disk, just parse protein.
        """
        self.protein_dict = self.initialize_protein(protein_seq)

    def run_validation(self, smiles_list: list) -> pd.DataFrame:
        """
        Legacy method that re-initializes SMILES each time.
        If you are preloading, prefer `run_preloaded_validation`.
        """
        # Re-init SMILES each time
        self.smiles_dict = self.initialize_smiles(smiles_list)

        # Create combined DataFrame
        screen_df = pd.DataFrame({
            'Protein': [p for p in self.protein_seq for _ in self.smiles_list],
            'Ligand':  [s for s in self.smiles_list for _ in self.protein_seq],
        })

        # Build dataset + loader
        dataset = ProteinMoleculeDataset(
            screen_df,
            self.smiles_dict,
            self.protein_dict,
            device=self.device
        )
        loader = DataLoader(
            dataset,
            batch_size=self.runtime_config.BATCH_SIZE,
            shuffle=False,
            follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
        )

        # Inference
        screen_df = virtual_screening(
            screen_df,
            self.model,
            loader,
            os.getcwd(),
            save_interpret=False,
            ligand_dict=self.smiles_dict,
            device=self.device,
            save_cluster=False,
        )
        return screen_df

    def run_preloaded_validation(self, combined_df: pd.DataFrame, preloaded_ligand_dict: dict) -> pd.DataFrame:
        """
        Evaluate binding affinity using a pre-built DataFrame of (Protein, Ligand)
        and a pre-featurized 'preloaded_ligand_dict' from initialize_smiles(...).
        This avoids re-initializing the SMILES each time.

        combined_df must have columns ['Protein', 'Ligand'].
        self.protein_dict must be set from run_challenge_start(...) or initialize_proteins_batch(...).
        """
        # Determine batch size based on data size and available memory
        batch_size = min(len(combined_df), self.runtime_config.BATCH_SIZE)
        
        dataset = ProteinMoleculeDataset(
            combined_df,
            preloaded_ligand_dict,
            self.protein_dict,
            device=self.device
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
        )

        with torch.no_grad():
            screen_df = virtual_screening(
                combined_df,
                self.model,
                loader,
                os.getcwd(),
                save_interpret=False,
                ligand_dict=preloaded_ligand_dict,
                device=self.device,
                save_cluster=False,
            )
        return screen_df

    def get_protein_by_sequence(self, sequence):
        """
        Helper method to find the index of a protein sequence in the batch.
        """
        try:
            index = self.protein_seq.index(sequence)
            return index
        except ValueError:
            return None
