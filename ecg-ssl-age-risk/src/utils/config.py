from pathlib import Path

class Config:
    # Project Root
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Data Paths
    # Detected from user context
    RAW_DATA_ROOT = Path(r"C:\Users\kbala\Documents\amrita\Sem 8\Code\Dataset\archive (2)\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    
    # Intermediate paths
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    
    # Dataset specific
    SAMPLING_RATE = 500  # Load at 500Hz
    TARGET_SAMPLING_RATE = 250 # Downsample to 250Hz for efficiency
    DURATION = 10  # seconds
    NUM_LEADS = 12
    
    # Label configuration
    # 5 diagnostic superclasses
    DIAGNOSTIC_SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Training Config
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # Windows often requires 0 workers for debugging, can bump to 4 later
    PIN_MEMORY = True
    
    # Model Config
    ENCODER_EMBED_DIM = 128
    
    @classmethod
    def get_ptbxl_csv_path(cls):
        return cls.RAW_DATA_ROOT / "ptbxl_database.csv"
        
    @classmethod
    def get_scp_statements_path(cls):
        return cls.RAW_DATA_ROOT / "scp_statements.csv"

# Global instance
config = Config()
