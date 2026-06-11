"""
Hugging Face Inference Endpoint Handler for HM-Conformer Deepfake Detection Model

This script handles inference requests for the HM-Conformer model deployed on
Hugging Face Inference Endpoints.

Expected input format:
    {
        "inputs": <audio_bytes> or <base64_encoded_audio>
    }

Output format:
    {
        "deepfake_score": <float>,  # Higher = more likely to be deepfake/spoof
        "is_bonafide": <bool>,      # True if likely real, False if likely fake
        "confidence": <float>       # Confidence score (0-1)
    }
"""

from typing import Dict, List, Any
import os
import sys
import io
import base64
import numpy as np
import torch
import soundfile as sf

# Add exp_lib to path
# Use absolute path relative to this file to ensure it works regardless of CWD
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_lib_path = os.path.join(current_dir, 'exp_lib')

# Check multiple potential locations
possible_lib_paths = [
    exp_lib_path,
    os.path.join(current_dir, 'HM-Conformer', 'exp_lib'),
    '/repository/exp_lib'  # Standard HF path
]

lib_added = False
for path in possible_lib_paths:
    if os.path.exists(path):
        if path not in sys.path:
            sys.path.append(path)
            print(f"Added {path} to sys.path")
            lib_added = True

try:
    import egg_exp
except ImportError as e:
    print(f"ERROR: Could not import egg_exp. Searched paths: {possible_lib_paths}")
    print(f"Current sys.path: {sys.path}")
    print(f"Contents of {current_dir}:")
    try:
        print(os.listdir(current_dir))
        if os.path.exists(exp_lib_path):
             print(f"Contents of {exp_lib_path}: {os.listdir(exp_lib_path)}")
    except Exception:
        pass
    
    # Try one last fallback if structure is different
    if os.path.exists(os.path.join(current_dir, 'HM-Conformer')):
         sys.path.insert(0, os.path.join(current_dir, 'HM-Conformer'))
         try:
             import egg_exp
         except ImportError:
             raise e
    else:
        raise e


class EndpointHandler:
    """Handler for HM-Conformer inference endpoint"""
    
    def __init__(self, path=""):
        """
        Initialize the model handler.
        
        Args:
            path: Path to model directory (default: current directory)
        """
        self.path = path if path else "."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration (from arguments.py defaults)
        self.config = {
            # Frontend model
            'bin_size': 120,
            'output_size': 128,
            'input_layer': "conv2d2",
            'pos_enc_layer_type': "rel_pos",
            'linear_units': 256,
            'cnn_module_kernel': 15,
            'dropout': 0.75,
            'emb_dropout': 0.3,
            
            # Backend model
            'use_pooling': False,
            'input_mean_std': False,
            'embedding_size': 64,
            
            # OCSoftmax loss
            'num_class': 1,
            'feat_dim': 2,
            'r_real': 0.9,
            'r_fake': 0.2,
            'alpha': 20.0,
            'loss_weight': [0.4, 0.3, 0.2, 0.1, 0.1],
            
            # Data processing
            'sample_rate': 16000,
            'n_lfcc': 40,
            'coef': 0.97,
            'n_fft': 512,
            'win_length': 320,
            'hop': 160,
            'with_delta': True,
            'with_emphasis': True,
            'with_energy': True,
            'test_crop_size': 16000 * 4,  # 4 seconds at 16kHz
            
            # Data augmentation (disabled for inference)
            'DA_wav_aug_list': [],  # No augmentation during inference
            'DA_frq_mask': False,   # No frequency masking during inference
        }
        
        # Build model framework
        self.framework = self._build_framework()
        
        # Load model weights
        self._load_model()
        
        # Set to evaluation mode
        self.framework.eval()
        
        print(f"Model loaded successfully on device: {self.device}")
    
    def _build_framework(self):
        """Build the HM-Conformer framework"""
        # Waveform augmentation (disabled for inference)
        augmentation = None
        
        # LFCC preprocessing
        preprocessing = egg_exp.framework.model.LFCC(
            self.config['sample_rate'],
            self.config['n_lfcc'],
            self.config['coef'],
            self.config['n_fft'],
            self.config['win_length'],
            self.config['hop'],
            self.config['with_delta'],
            self.config['with_emphasis'],
            self.config['with_energy'],
            self.config['DA_frq_mask'],
            0.5,  # p (not used if DA_frq_mask=False)
            20    # max (not used if DA_frq_mask=False)
        )
        
        # Frontend
        frontend = egg_exp.framework.model.HM_Conformer(
            bin_size=self.config['bin_size'],
            output_size=self.config['output_size'],
            input_layer=self.config['input_layer'],
            pos_enc_layer_type=self.config['pos_enc_layer_type'],
            linear_units=self.config['linear_units'],
            cnn_module_kernel=self.config['cnn_module_kernel'],
            dropout=self.config['dropout'],
            emb_dropout=self.config['emb_dropout'],
            multiloss=True
        )
        
        # Backends (5 backends)
        backends = []
        for i in range(5):
            backend = egg_exp.framework.model.CLSBackend(
                in_dim=self.config['output_size'],
                hidden_dim=self.config['embedding_size'],
                use_pooling=self.config['use_pooling'],
                input_mean_std=self.config['input_mean_std']
            )
            backends.append(backend)
        
        # Losses (5 OCSoftmax losses)
        criterions = []
        for i in range(5):
            criterion = egg_exp.framework.loss.OCSoftmax(
                embedding_size=self.config['embedding_size'],
                num_class=self.config['num_class'],
                feat_dim=self.config['feat_dim'],
                r_real=self.config['r_real'],
                r_fake=self.config['r_fake'],
                alpha=self.config['alpha']
            )
            criterions.append(criterion)
        
        # Create framework
        framework = egg_exp.framework.DeepfakeDetectionFramework_DA_multiloss(
            augmentation=augmentation,
            preprocessing=preprocessing,
            frontend=frontend,
            backend=backends,
            loss=criterions,
            loss_weight=self.config['loss_weight']
        )
        
        # Move to device
        framework.device = self.device
        for module in framework.trainable_modules.values():
            module.to(self.device)
        
        return framework
    
    def _load_model(self):
        """Load model checkpoints"""
        # Expected checkpoint filenames
        checkpoint_files = {
            'frontend': 'check_point_DF_frontend_20.pt',
            'backend0': 'check_point_DF_backend0_20.pt',
            'backend1': 'check_point_DF_backend1_20.pt',
            'backend2': 'check_point_DF_backend2_20.pt',
            'backend3': 'check_point_DF_backend3_20.pt',
            'backend4': 'check_point_DF_backend4_20.pt',
            'loss0': 'check_point_DF_loss0_20.pt',
            'loss1': 'check_point_DF_loss1_20.pt',
            'loss2': 'check_point_DF_loss2_20.pt',
            'loss3': 'check_point_DF_loss3_20.pt',
            'loss4': 'check_point_DF_loss4_20.pt',
        }
        
        # Try different possible paths
        possible_paths = [
            os.path.join(self.path, 'params'),
            os.path.join(self.path, 'HM-Conformer', 'hm_conformer', 'params'),
            os.path.join(self.path, 'hm_conformer', 'params'),
            './params',
            './HM-Conformer/hm_conformer/params',
        ]
        
        params_path = None
        for p in possible_paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, checkpoint_files['frontend'])):
                params_path = p
                break
        
        if params_path is None:
            raise FileNotFoundError(
                f"Could not find model checkpoints. Searched in: {possible_paths}\n"
                f"Please ensure checkpoints are in one of these locations."
            )
        
        print(f"Loading checkpoints from: {params_path}")
        
        # Load each checkpoint
        for module_name, filename in checkpoint_files.items():
            checkpoint_path = os.path.join(params_path, filename)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                model = self.framework.trainable_modules[module_name]
                
                # Handle 'module.' prefix from DDP
                model_state = model.state_dict()
                loaded_state = {}
                for name, param in state_dict.items():
                    if name not in model_state:
                        # Try removing 'module.' prefix
                        if name.startswith('module.'):
                            name = name[7:]
                    if name in model_state:
                        if model_state[name].size() == param.size():
                            loaded_state[name] = param
                        else:
                            print(f"Warning: Size mismatch for {module_name}.{name}: "
                                  f"model={model_state[name].size()}, loaded={param.size()}")
                    else:
                        print(f"Warning: Parameter {module_name}.{name} not found in model")
                
                model.load_state_dict(loaded_state, strict=False)
                print(f"Loaded {module_name}")
            except Exception as e:
                print(f"Error loading {module_name}: {e}")
                raise
    
    def _process_audio(self, audio_input):
        """
        Process audio input to model format.
        
        Args:
            audio_input: Can be bytes, base64 string, or file path
            
        Returns:
            numpy array: Audio waveform (1D, 64000 samples = 4 seconds at 16kHz)
        """
        # Handle different input formats
        if isinstance(audio_input, str):
            # Try base64 decode first
            try:
                audio_bytes = base64.b64decode(audio_input)
                audio_input = io.BytesIO(audio_bytes)
            except (base64.binascii.Error, TypeError):
                # Try as file path
                if os.path.exists(audio_input):
                    audio_input = audio_input
                else:
                    raise ValueError(f"Could not decode base64 or find file: {audio_input[:50]}...")
        
        elif isinstance(audio_input, bytes):
            audio_input = io.BytesIO(audio_input)
        
        elif isinstance(audio_input, np.ndarray):
            # Already processed numpy array
            wav = audio_input
            # Ensure it's 1D
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            # Resample if needed (will be handled below)
            sr = self.config['sample_rate']  # Assume correct sample rate for numpy input
            if len(wav) != self.config['test_crop_size']:
                # Crop/pad
                crop_size = self.config['test_crop_size']
                if len(wav) < crop_size:
                    wav = np.pad(wav, (0, crop_size - len(wav)), 'wrap')
                else:
                    wav = wav[:crop_size]
            return wav.astype(np.float32)
        
        # Load audio using soundfile
        try:
            wav, sr = sf.read(audio_input)
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")
        
        # Convert to mono if stereo
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        
        # Resample to 16kHz if needed
        if sr != self.config['sample_rate']:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.config['sample_rate'])
        
        # Crop/pad to exactly 4 seconds (64000 samples)
        crop_size = self.config['test_crop_size']
        if len(wav) < crop_size:
            # Pad by wrapping
            shortage = crop_size - len(wav)
            wav = np.pad(wav, (0, shortage), 'wrap')
        else:
            # Truncate to first 4 seconds
            wav = wav[:crop_size]
        
        return wav.astype(np.float32)
    
    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle inference request.
        
        Args:
            data: Dictionary with 'inputs' key containing audio data
            
        Returns:
            List of dictionaries with deepfake detection results
        """
        try:
            # Extract audio input
            if 'inputs' not in data:
                # Some endpoints might pass data directly if not JSON wrapped?
                # But standard is {"inputs": ...}
                # Let's check if 'body' or raw data is passed in some cases, 
                # but typically 'inputs' is the key.
                if 'body' in data:
                     # fallback for some raw requests
                     pass
                # raise ValueError("Input must contain 'inputs' key with audio data")
                # If inputs is missing, maybe the data itself is the input? 
                # But let's stick to the current logic unless we see issues.
                pass

            audio_input = data.get('inputs', data)

            # Process audio
            wav = self._process_audio(audio_input)
            
            # Convert to tensor and add batch dimension
            wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device)  # Shape: (1, 64000)
            
            # Run inference
            with torch.no_grad():
                score = self.framework(wav_tensor)  # Returns score tensor
                
                # Extract score value
                if score.size(1) == 1:
                    deepfake_score = score[0, 0].item()
                else:
                    deepfake_score = score[0, 1].item()
            
            # Interpret score
            # OCSoftmax returns: 1 - cosine_similarity
            # Higher score = more likely to be deepfake/spoof
            # Lower score = more likely to be bonafide/real
            
            # Determine if bonafide (real) or spoof (fake)
            # Threshold can be adjusted based on your EER threshold
            threshold = 0.5  # Default threshold
            is_bonafide = deepfake_score < threshold
            
            # Confidence is distance from threshold
            confidence = abs(deepfake_score - threshold) * 2
            confidence = max(0.0, min(1.0, confidence))
            
            return [{
                "label": "bonafide" if is_bonafide else "spoof",
                "score": float(deepfake_score),  # HF usually expects 'score'
                "deepfake_score": float(deepfake_score),
                "is_bonafide": bool(is_bonafide),
                "confidence": float(confidence),
                "threshold_used": float(threshold)
            }]
            
        except Exception as e:
            return [{
                "error": str(e),
                "error_type": type(e).__name__
            }]

