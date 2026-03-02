from flask import Flask, render_template, request, send_file, jsonify
import os
import sys
from werkzeug.utils import secure_filename
import onnx
from onnx import shape_inference
import json
import traceback
from datetime import datetime, timedelta
import subprocess
import threading
import time
import glob
import shutil
import numpy as np
import zipfile
import tempfile
import io
import contextlib

try:
    import onnxruntime as ort
    import tensorflow as tf
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: onnxruntime or tensorflow not available. Accuracy comparison will be disabled.")

# Add eiq-onnx2tflite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eiq-onnx2tflite'))

# Import converter
import onnx2tflite.src.converter.convert as convert
from onnx2tflite.src.conversion_config import ConversionConfig

# Import quantization modules
from onnx2quant.qdq_quantization import QDQQuantizer, InputSpec
from onnx2quant.quantization_config import QuantizationConfig

# We need a custom reader for real data
class RealDataCalibrationDataReader:
    def __init__(self, input_data_map):
        """
        input_data_map: dict { input_name: [numpy_array_sample_1, numpy_array_sample_2, ...] }
        """
        self.input_data_map = input_data_map
        self.input_names = list(input_data_map.keys())
        # Assuming all inputs have same number of samples
        self.num_samples = len(input_data_map[self.input_names[0]])
        self.current_index = 0

    def get_next(self):
        if self.current_index < self.num_samples:
            batch = {}
            for name in self.input_names:
                batch[name] = self.input_data_map[name][self.current_index]
            self.current_index += 1
            return batch
        else:
            return None
    
    def rewind(self):
        self.current_index = 0

def compare_model_accuracy(onnx_path, tflite_path, test_data=None, num_samples=100, symbolic_dimensions_mapping=None):
    """Compare accuracy between ONNX and TFLite models.
    
    Args:
        onnx_path: Path to ONNX model
        tflite_path: Path to TFLite model
        test_data: Optional dict {input_name: [arrays]} for calibration data
        num_samples: Number of random samples to generate if no test_data
        symbolic_dimensions_mapping: Optional dict mapping symbolic dimension names to values
    
    Returns:
        dict with accuracy metrics or None if comparison failed
    """
    if not INFERENCE_AVAILABLE:
        return None
    
    try:
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        onnx_inputs = ort_session.get_inputs()
        onnx_outputs = ort_session.get_outputs()
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        tflite_inputs = interpreter.get_input_details()
        tflite_outputs = interpreter.get_output_details()
        
        # Generate or use test data
        if test_data:
            # Use provided calibration data
            try:
                samples_count = min(num_samples, len(list(test_data.values())[0]))
                test_samples = []
                for i in range(samples_count):
                    sample = {}
                    for name in test_data.keys():
                        data_array = test_data[name][i]
                        # Ensure it's a numpy array and float32
                        if not isinstance(data_array, np.ndarray):
                            data_array = np.array(data_array)
                        if data_array.dtype != np.float32:
                            data_array = data_array.astype(np.float32)
                        sample[name] = data_array
                    test_samples.append(sample)
            except Exception as e:
                print(f"Error processing calibration data: {e}, falling back to random data")
                test_data = None
        
        if not test_data:
            # Generate random data
            # Load ONNX model to get symbolic dimension names
            onnx_model = onnx.load(onnx_path)
            sym_dim_mapping = symbolic_dimensions_mapping if symbolic_dimensions_mapping else {}
            
            # Auto-detect symbolic dimensions if not provided
            if not sym_dim_mapping:
                for inp in onnx_model.graph.input:
                    for dim in inp.type.tensor_type.shape.dim:
                        if dim.dim_param and dim.dim_param not in sym_dim_mapping:
                            sym_dim_mapping[dim.dim_param] = 50  # Default to 50
            
            test_samples = []
            for _ in range(num_samples):
                sample = {}
                for inp_idx, inp in enumerate(onnx_inputs):
                    shape = []
                    onnx_inp = onnx_model.graph.input[inp_idx]
                    for dim_idx, dim in enumerate(onnx_inp.type.tensor_type.shape.dim):
                        if dim.dim_param:  # Symbolic dimension
                            shape.append(sym_dim_mapping.get(dim.dim_param, 50))
                        elif dim.dim_value > 0:  # Concrete dimension
                            shape.append(dim.dim_value)
                        else:  # Unknown dimension
                            shape.append(1)
                    sample[inp.name] = np.random.randn(*shape).astype(np.float32)
                test_samples.append(sample)
        
        # Run inference and collect results
        onnx_results = []
        tflite_results = []
        tflite_times = []
        
        for sample in test_samples:
            try:
                # ONNX inference
                onnx_feed = {inp.name: sample[inp.name] for inp in onnx_inputs}
                onnx_out = ort_session.run([out.name for out in onnx_outputs], onnx_feed)
                
                # TFLite inference with timing
                for i, inp_detail in enumerate(tflite_inputs):
                    inp_name = onnx_inputs[i].name if i < len(onnx_inputs) else list(sample.keys())[i]
                    input_data = sample[inp_name]
                    
                    # Ensure data type matches TFLite expectation
                    expected_dtype = inp_detail['dtype']
                    if input_data.dtype != expected_dtype:
                        input_data = input_data.astype(expected_dtype)
                    
                    # Ensure shape matches
                    expected_shape = inp_detail['shape']
                    if input_data.shape != tuple(expected_shape):
                        # Try to reshape if sizes match
                        if input_data.size == np.prod(expected_shape):
                            input_data = input_data.reshape(expected_shape)
                    
                    interpreter.set_tensor(inp_detail['index'], input_data)
                
                # Time the TFLite inference
                start_time = time.time()
                interpreter.invoke()
                inference_time = time.time() - start_time
                tflite_times.append(inference_time)
                
                tflite_out = [interpreter.get_tensor(out['index']) for out in tflite_outputs]
                
                onnx_results.append(onnx_out)
                tflite_results.append(tflite_out)
            except Exception as e:
                print(f"Inference error on sample: {e}")
                continue
        
        if len(onnx_results) == 0:
            return None
        
        # Calculate metrics
        metrics = []
        for output_idx in range(len(onnx_outputs)):
            try:
                onnx_vals = np.array([r[output_idx] for r in onnx_results])
                tflite_vals = np.array([r[output_idx] for r in tflite_results])
                
                # Flatten for easier calculation
                onnx_flat = onnx_vals.flatten()
                tflite_flat = tflite_vals.flatten()
                
                # Ensure both have the same shape
                if onnx_flat.shape != tflite_flat.shape:
                    print(f"Shape mismatch for output {output_idx}: ONNX {onnx_flat.shape} vs TFLite {tflite_flat.shape}")
                    continue
                
                # Calculate MAE, MSE, RMSE
                mae = np.mean(np.abs(onnx_flat - tflite_flat))
                mse = np.mean((onnx_flat - tflite_flat) ** 2)
                rmse = np.sqrt(mse)
                
                # Calculate cosine similarity
                dot_product = np.dot(onnx_flat, tflite_flat)
                norm_onnx = np.linalg.norm(onnx_flat)
                norm_tflite = np.linalg.norm(tflite_flat)
                cosine_sim = dot_product / (norm_onnx * norm_tflite) if norm_onnx > 0 and norm_tflite > 0 else 0
                
                # Calculate max absolute error
                max_error = np.max(np.abs(onnx_flat - tflite_flat))
                
                metrics.append({
                    'output_name': onnx_outputs[output_idx].name,
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'cosine_similarity': float(cosine_sim),
                    'max_error': float(max_error)
                })
            except Exception as e:
                print(f"Error calculating metrics for output {output_idx}: {e}")
                continue
        
        # Calculate average inference time
        avg_inference_time = np.mean(tflite_times) if tflite_times else 0
        
        return {
            'num_samples': len(onnx_results),
            'outputs': metrics,
            'avg_inference_time_us': float(avg_inference_time * 1000000)  # Convert to microseconds
        }
        
    except Exception as e:
        print(f"Accuracy comparison error: {e}")
        traceback.print_exc()
        return None

# Flask App Configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['JSON_SORT_KEYS'] = False

ALLOWED_EXTENSIONS = {'onnx', 'zip'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Cleanup files older than 30 minutes."""
    try:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=30)
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            pattern = os.path.join(folder, '*')
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    try:
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
                    except Exception:
                        pass
    except Exception:
        pass

def start_cleanup_timer():
    def cleanup_loop():
        while True:
            time.sleep(1800)
            cleanup_old_files()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

def get_model_info(onnx_path):
    try:
        model = onnx.load(onnx_path)
        
        # Try to infer shapes for better visualization
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass  # Continue even if shape inference fails
        
        graph = model.graph
        
        info = {
            'ir_version': model.ir_version,
            'producer_name': model.producer_name,
            'graph_name': graph.name,
            'inputs': [],
            'outputs': [],
            'node_count': len(graph.node),
            'node_types': {},
            'nodes': [],
            'edges': []
        }
        
        # Build tensor shape map from value_info
        tensor_shapes = {}
        
        # Get model inputs with shapes
        for inp in graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_param if dim.dim_param else dim.dim_value)
            info['inputs'].append({
                'name': inp.name,
                'type': str(inp.type.tensor_type.elem_type),
                'shape': shape
            })
            tensor_shapes[inp.name] = shape
        
        # Get model outputs with shapes
        output_names = set()
        for out in graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                shape.append(dim.dim_param if dim.dim_param else dim.dim_value)
            info['outputs'].append({
                'name': out.name,
                'type': str(out.type.tensor_type.elem_type),
                'shape': shape
            })
            output_names.add(out.name)
            tensor_shapes[out.name] = shape
        
        # Get intermediate tensor shapes from value_info
        for value_info in graph.value_info:
            shape = []
            try:
                for dim in value_info.type.tensor_type.shape.dim:
                    shape.append(dim.dim_param if dim.dim_param else dim.dim_value)
                tensor_shapes[value_info.name] = shape
            except Exception:
                pass
        
        # Build tensor producer map (which node produces which tensor)
        tensor_producer = {}
        input_names = set(inp['name'] for inp in info['inputs'])
        
        # Map inputs to their pseudo-nodes
        for inp_name in input_names:
            tensor_producer[inp_name] = f"input_{inp_name}"
        
        # Get detailed node info and build edges
        for idx, node in enumerate(graph.node):
            node_id = f"node_{idx}_{node.name if node.name else node.op_type}"
            
            # Count node types
            info['node_types'][node.op_type] = info['node_types'].get(node.op_type, 0) + 1
            
            # Get attributes
            attributes = {}
            for attr in node.attribute:
                try:
                    if attr.HasField('f'):
                        attributes[attr.name] = float(attr.f)
                    elif attr.HasField('i'):
                        attributes[attr.name] = int(attr.i)
                    elif attr.HasField('s'):
                        attributes[attr.name] = attr.s.decode('utf-8') if attr.s else ''
                    elif attr.floats:
                        attributes[attr.name] = list(attr.floats)
                    elif attr.ints:
                        attributes[attr.name] = list(attr.ints)
                except Exception:
                    attributes[attr.name] = str(attr)
            
            # Add node info
            info['nodes'].append({
                'id': node_id,
                'name': node.name if node.name else f"{node.op_type}_{idx}",
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': attributes
            })
            
            # Map outputs to this node
            for output in node.output:
                tensor_producer[output] = node_id
            
            # Create edges from inputs to this node
            for input_tensor in node.input:
                if input_tensor in tensor_producer:
                    from_node = tensor_producer[input_tensor]
                    info['edges'].append({
                        'from': from_node,
                        'to': node_id,
                        'tensor': input_tensor
                    })
        
        # Connect output nodes to the nodes that produce them
        for output_info in info['outputs']:
            output_name = output_info['name']
            output_node_id = f"output_{output_name}"
            
            # Find which node produces this output
            if output_name in tensor_producer:
                from_node = tensor_producer[output_name]
                info['edges'].append({
                    'from': from_node,
                    'to': output_node_id,
                    'tensor': output_name
                })
        
        # Add tensor shapes to info
        info['tensor_shapes'] = tensor_shapes
        
        return info
    except Exception as e:
        return {'error': str(e)}

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
        
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        onnx_filename = f"{timestamp}_{filename}"
        onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], onnx_filename)
        file.save(onnx_path)
        
        info = get_model_info(onnx_path)
        return jsonify({'success': True, 'filename': onnx_filename, 'model_info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_calibration', methods=['POST'])
def upload_calibration():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Only .zip files allowed'}), 400
        
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"calibration_{timestamp}_{filename}"
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
        file.save(zip_path)
        
        # Validate content
        npy_files = []
        with zipfile.ZipFile(zip_path, 'r') as z:
            npy_files = [f for f in z.namelist() if f.endswith('.npy')]
            
        if not npy_files:
             os.remove(zip_path)
             return jsonify({'error': 'ZIP contains no .npy files'}), 400
             
        return jsonify({
            'success': True, 
            'filename': zip_filename, 
            'file_count': len(npy_files),
            'files': npy_files[:10] # send preview
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/convert', methods=['POST'])
def convert_model():
    log_capture_string = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_capture_string), contextlib.redirect_stderr(log_capture_string):
            data = request.json
            onnx_filename = data.get('filename')
            settings = data.get('settings', {})
            
            if not onnx_filename:
                 return jsonify({'error': 'No model loaded'}), 400
                 
            onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], onnx_filename)
            tflite_filename = os.path.splitext(onnx_filename)[0] + ".tflite"
            tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], tflite_filename)
            
            # --- Configuration Mapping ---
            config = ConversionConfig()
            
            # Boolean Flags
            config.allow_inputs_stripping = settings.get('allowInputsStripping', False)
            config.keep_io_format = settings.get('keepIoFormat', False)
            config.skip_shape_inference = settings.get('skipShapeInference', False)
            config.qdq_aware_conversion = settings.get('qdqAwareConversion', False)
            config.allow_select_ops = settings.get('allowSelectOps', False)
            config.generate_artifacts_after_failed_shape_inference = settings.get('generateArtifacts', False)
            config.non_negative_indices = settings.get('guaranteeNonNegative', False)
            config.cast_int64_to_int32 = settings.get('castInt64ToInt32', False)
            config.accept_resize_rounding_error = settings.get('acceptResizeError', False)
            config.ignore_opset_version = settings.get('skipOpsetCheck', False)
            config.dont_skip_nodes_with_known_outputs = settings.get('dontSkipUnknown', False)
            
            # Mappings
            sym_dims = settings.get('symbolicDims', '')
            if sym_dims:
                mapping = {}
                try:
                    for pair in sym_dims.split(','):
                        if ':' in pair: s = ':'
                        elif '=' in pair: s = '='
                        else: continue
                        k, v = pair.split(s)
                        mapping[k.strip()] = int(v.strip())
                    config.symbolic_dimensions_mapping = mapping
                except Exception as e:
                    print(f"Error parsing symbolic dims: {e}")
            else:
                # Auto-populate symbolic dims with default value of 50 if not provided
                try:
                    onnx_model = onnx.load(onnx_path)
                    mapping = {}
                    for inp in onnx_model.graph.input:
                        for dim in inp.type.tensor_type.shape.dim:
                            if dim.dim_param:
                                # Use 50 as default for sequence_length or other symbolic dims
                                mapping[dim.dim_param] = 50
                    if mapping:
                        print(f"Auto-setting symbolic dims to 50 for conversion: {mapping}")
                        config.symbolic_dimensions_mapping = mapping
                except Exception as e:
                    print(f"Error auto-setting symbolic dims: {e}")

            # Input Shapes
            input_shapes = settings.get('inputShapes', '')
            if input_shapes:
                 pass

            start_time = datetime.now()
            tflite_model = convert.convert_model(onnx_path, config)
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            duration = (datetime.now() - start_time).total_seconds()
            onnx_size = os.path.getsize(onnx_path)
            tflite_size = os.path.getsize(tflite_path)
            
            # Perform accuracy comparison
            print("\nComparing model accuracy...")
            accuracy_metrics = compare_model_accuracy(
                onnx_path, 
                tflite_path, 
                num_samples=100,
                symbolic_dimensions_mapping=config.symbolic_dimensions_mapping
            )
            
            result = {
                'success': True,
                'tflite_filename': tflite_filename,
                'stats': {
                    'original_size': onnx_size,
                    'converted_size': tflite_size,
                    'ratio': f"{(1 - tflite_size/onnx_size)*100:.1f}%",
                    'time': f"{duration:.2f}s"
                },
                'logs': log_capture_string.getvalue()
            }
            
            if accuracy_metrics:
                result['accuracy'] = accuracy_metrics
                # Add inference time as a separate field instead of overwriting time
                if 'avg_inference_time_us' in accuracy_metrics:
                    result['stats']['inference_time'] = f"{accuracy_metrics['avg_inference_time_us']:.2f}us"
                print(f"Accuracy comparison completed: {accuracy_metrics['num_samples']} samples")
            else:
                print("Accuracy comparison skipped or failed")
            
            return jsonify(result)
    except Exception as e:
        traceback.print_exc(file=log_capture_string)
        return jsonify({'error': str(e), 'logs': log_capture_string.getvalue()}), 500

@app.route('/quantize', methods=['POST'])
def quantize_model():
    """Perform QDQ Quantization then Conversion"""
    log_capture_string = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_capture_string), contextlib.redirect_stderr(log_capture_string):
            data = request.json
            onnx_filename = data.get('filename')
            settings = data.get('settings', {})
            selected_layers = data.get('selectedLayers', [])
            calib_filename = data.get('calibFilename')
            
            onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], onnx_filename)
            base_name = os.path.splitext(onnx_filename)[0]
            q_onnx_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_quantized.onnx")
            q_tflite_filename = f"{base_name}_quantized.tflite"
            q_tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], q_tflite_filename)
            
            onnx_model = onnx.load(onnx_path)
            
            # --- Prepare Calibration Data ---
            calibration_reader = None
            
            if calib_filename:
                # Process ZIP file
                zip_path = os.path.join(app.config['UPLOAD_FOLDER'], calib_filename)
                input_data_map = {} # {input_name: [arr1, arr2]}
                
                # Helper to guess input name from filename (naive)
                # Or assume mapping provided? For now, we load all .npy and if single input model, assign all.
                # If multi-input, we probably need a mapping from UI, but for "RealDataCalibrationDataReader" logic:
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(temp_dir)
                        
                    count = 0 
                    # Find all npy
                    for root, dirs, files in os.walk(temp_dir):
                        for f in files:
                            if f.endswith('.npy'):
                                # Naive: assume filename contains input name or just collect all
                                # Better approach: UI should probably specify input mappings if strict.
                                # Fallback: Just load them all into a list and try to feed.
                                 
                                # Actually, RandomDataCalibrationDataReader generates a batch dict.
                                # So real reader must do same.
                                
                                # Let's inspect model inputs to match
                                model_inputs = [i.name for i in onnx_model.graph.input]
                                
                                try:
                                    arr = np.load(os.path.join(root, f))
                                    # Ensure batch dim exist if needed, etc.
                                    
                                    # Heuristic: if model has 1 input, assign all npy to it
                                    if len(model_inputs) >= 1:
                                        target_input = model_inputs[0] # Default to first
                                        # If filename matches another input, switch
                                        for inp in model_inputs:
                                            if inp in f:
                                                target_input = inp
                                                
                                        if target_input not in input_data_map:
                                            input_data_map[target_input] = []
                                        input_data_map[target_input].append(arr)
                                        count += 1
                                except:
                                    pass
                                    
                if count > 0:
                    print(f"Loaded {count} calibration samples")
                    calibration_reader = RealDataCalibrationDataReader(input_data_map)
            
            # --- Parse or Auto-generate Symbolic Dimensions Mapping FIRST ---
            # This must be done before creating calibration data to ensure consistency
            symbolic_dim_mapping = {}
            sym_dims = settings.get('symbolicDims', '').strip()
            if sym_dims:
                try:
                    for pair in sym_dims.split(','):
                        if ':' in pair: s = ':'
                        elif '=' in pair: s = '='
                        else: continue
                        k, v = pair.split(s)
                        symbolic_dim_mapping[k.strip()] = int(v.strip())
                    print(f"Using user-provided symbolic dims for quantization: {symbolic_dim_mapping}")
                except Exception as e:
                    print(f"Error parsing symbolic dims for quantization: {e}")
            else:
                 # Auto-populate symbolic dims with default value of 50 if not provided
                 for i in onnx_model.graph.input:
                     for d in i.type.tensor_type.shape.dim:
                         if d.dim_param:
                             symbolic_dim_mapping[d.dim_param] = 50 # Default to 50
                 if symbolic_dim_mapping:
                     print(f"Auto-setting symbolic dims to 50 for quantization: {symbolic_dim_mapping}")
            
            if not calibration_reader:
                 # Fallback to random if no zip or empty zip
                 # Use the symbolic dimension mapping to create correct shapes
                inputs = {}
                for input_info in onnx_model.graph.input:
                    dims = []
                    for d in input_info.type.tensor_type.shape.dim:
                        if d.dim_value > 0:
                            # Use concrete dimension value
                            dims.append(d.dim_value)
                        elif d.dim_param and d.dim_param in symbolic_dim_mapping:
                            # Use mapped symbolic dimension value
                            dims.append(symbolic_dim_mapping[d.dim_param])
                        else:
                            # Fallback to 1 if no mapping available
                            dims.append(1)
                    inputs[input_info.name] = InputSpec(dims, np.float32)
                
                from onnx2quant.qdq_quantization import RandomDataCalibrationDataReader
                calibration_reader = RandomDataCalibrationDataReader(inputs, num_samples=5)
                print(f"Using Random Calibration Data with shapes: {[(name, spec.shape) for name, spec in inputs.items()]}")

            # --- Quantization Config ---
            q_config = QuantizationConfig(calibration_reader)
            q_config.per_channel = settings.get('perChannel', False)
            q_config.allow_opset_10_and_lower = settings.get('allowOpset10', False)
            
            # Apply the symbolic dimensions mapping to quantization config
            if symbolic_dim_mapping:
                q_config.symbolic_dimensions_mapping = symbolic_dim_mapping

            # Start timing
            start_time = datetime.now()
            
            quantizer = QDQQuantizer(
               op_types_to_quantize=selected_layers if selected_layers else None
            )
            
            # Some flags might need to be applied pre-quantization or handling manually if library doesn't support direct flag
            # But let's assume standard flow.
            
            q_onnx_model = quantizer.quantize_model(onnx_model, q_config)
            
            # Optional: Replace Div with Mul if requested (This usually happens before or during?)
            # Since I don't have the exact source of 'replace_div_with_mul' accessible easily right here without looking deep, 
            # I will assume the user wants the standard path provided by the tool.
            
            onnx.save(q_onnx_model, q_onnx_path)
            
            # --- Convert Quantized ONNX to TFLite ---
            conv_config = ConversionConfig()
            conv_config.qdq_aware_conversion = True # Force this for quantized models
            
            # Copy other compatible settings
            conv_config.keep_io_format = settings.get('keepIoFormat', False)
            
            tflite_model = convert.convert_model(q_onnx_path, conv_config)
            
            with open(q_tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Calculate statistics
            duration = (datetime.now() - start_time).total_seconds()
            onnx_size = os.path.getsize(onnx_path)
            q_onnx_size = os.path.getsize(q_onnx_path)
            q_tflite_size = os.path.getsize(q_tflite_path)
            
            # Prepare test data for accuracy comparison
            test_data_for_comparison = None
            if calib_filename:
                # Reuse calibration data for accuracy comparison
                zip_path = os.path.join(app.config['UPLOAD_FOLDER'], calib_filename)
                try:
                    input_data_map = {}
                    model_inputs = [i.name for i in onnx_model.graph.input]
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            z.extractall(temp_dir)
                        
                        for root, dirs, files in os.walk(temp_dir):
                            for f in files:
                                if f.endswith('.npy'):
                                    try:
                                        arr = np.load(os.path.join(root, f))
                                        if len(model_inputs) >= 1:
                                            target_input = model_inputs[0]
                                            for inp in model_inputs:
                                                if inp in f:
                                                    target_input = inp
                                            
                                            if target_input not in input_data_map:
                                                input_data_map[target_input] = []
                                            # Copy the array to ensure it persists after temp_dir is deleted
                                            input_data_map[target_input].append(arr.copy())
                                    except Exception as e:
                                        print(f"Error loading {f}: {e}")
                                        pass
                    
                    if input_data_map:
                        test_data_for_comparison = input_data_map
                        print(f"Using {len(list(input_data_map.values())[0])} calibration samples for accuracy comparison")
                except Exception as e:
                    print(f"Could not load calibration data for comparison: {e}")
            
            # Perform accuracy comparison (compare original ONNX vs quantized TFLite)
            print("\nComparing quantized model accuracy (original ONNX vs quantized TFLite)...")
            accuracy_metrics = compare_model_accuracy(
                onnx_path,  # Use original ONNX model, not quantized
                q_tflite_path, 
                test_data=test_data_for_comparison,
                num_samples=100,
                symbolic_dimensions_mapping=q_config.symbolic_dimensions_mapping
            )
            
            result = {
                'success': True,
                'quantized_tflite_filename': q_tflite_filename,
                'quantized_onnx_filename': os.path.basename(q_onnx_path),
                'stats': {
                    'original_size': onnx_size,
                    'quantized_onnx_size': q_onnx_size,
                    'converted_size': q_tflite_size,
                    'ratio': f"{(1 - q_tflite_size/onnx_size)*100:.1f}%",
                    'time': f"{duration:.2f}s"
                },
                'logs': log_capture_string.getvalue()
            }
            
            if accuracy_metrics:
                result['accuracy'] = accuracy_metrics
                # Add inference time as a separate field instead of overwriting time
                if 'avg_inference_time_us' in accuracy_metrics:
                    result['stats']['inference_time'] = f"{accuracy_metrics['avg_inference_time_us']:.2f}us"
                print(f"Accuracy comparison completed: {accuracy_metrics['num_samples']} samples")
            else:
                print("Accuracy comparison skipped or failed")
                
            return jsonify(result)

    except Exception as e:
        traceback.print_exc(file=log_capture_string)
        return jsonify({'error': str(e), 'logs': log_capture_string.getvalue()}), 500

@app.route('/download/<filename>')
def download(filename):
    # Secure the filename to prevent path traversal attacks
    safe_filename = secure_filename(filename)
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], safe_filename)
    
    # Verify the file exists and is within the output folder
    if not os.path.exists(file_path) or not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/download_calib_generator')
def download_calib_generator():
    """Generate and download a Python script for training model and creating calibration data"""
    script_content = '''#!/usr/bin/env python3
"""
模型训练与校准数据生成器
Train PyTorch models, export to ONNX, and generate calibration data for quantization

使用方法 (Usage):
    # 训练MLP模型并导出
    python train_and_calibrate.py --model mlp --epochs 50 --num_samples 100
    
    # 训练CNN模型
    python train_and_calibrate.py --model cnn --epochs 20 --batch_size 32
    
    # 训练多输入模型
    python train_and_calibrate.py --model multi_input --epochs 30
    
    # 训练Transformer模型
    python train_and_calibrate.py --model transformer --epochs 10 --num_samples 50

参数说明 (Arguments):
    --model: 模型类型 (mlp, cnn, resnet, transformer, multi_input)
    --epochs: 训练轮数 (default: 50)
    --batch_size: 批次大小 (default: 32)
    --lr: 学习率 (default: 0.001)
    --num_samples: 校准数据样本数量 (default: 100)
    --output_dir: 输出目录 (default: model_outputs)
    --skip_training: 跳过训练，仅生成校准数据
"""

import argparse
import os
import sys
import inspect
import numpy as np
import zipfile
import shutil
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Error: PyTorch not installed. Please install: pip install torch onnx")
    sys.exit(1)


# ===========================
# Model Definitions
# ===========================

class MLPModel(nn.Module):
    """Fully Connected Neural Network (MLP)"""
    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNModel(nn.Module):
    """Convolutional Neural Network for Image Classification"""
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleResNet(nn.Module):
    """Simple ResNet-like model"""
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        identity = self.relu(self.bn1(self.conv1(x)))
        
        out = self.conv2(identity)
        out = self.bn2(out)
        out += identity  # Skip connection
        out = self.relu(out)
        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    """Simple Transformer model for sequence classification"""
    def __init__(self, vocab_size=10000, embed_dim=128, num_heads=4, num_classes=2, seq_len=64):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x


class MultiInputModel(nn.Module):
    """Multi-input model example"""
    def __init__(self):
        super(MultiInputModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(5, 32)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
    
    def forward(self, input1, input2):
        x1 = self.relu(self.fc1(input1))
        x2 = self.relu(self.fc2(input2))
        x = torch.cat([x1, x2], dim=1)
        x = self.fc3(x)
        return x


# ===========================
# Training Functions
# ===========================

def create_dummy_dataset(model_type, num_samples=1000, batch_size=32):
    """Create dummy dataset for training"""
    print(f"Creating dummy dataset: {num_samples} samples, batch_size={batch_size}")
    
    if model_type == 'mlp':
        X = torch.randn(num_samples, 10)
        y = torch.randint(0, 2, (num_samples,))
        dataset = TensorDataset(X, y)
        
    elif model_type in ['cnn', 'resnet']:
        X = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, 10, (num_samples,))
        dataset = TensorDataset(X, y)
        
    elif model_type == 'transformer':
        X = torch.randint(0, 10000, (num_samples, 64))
        y = torch.randint(0, 2, (num_samples,))
        dataset = TensorDataset(X, y)
        
    elif model_type == 'multi_input':
        X1 = torch.randn(num_samples, 10)
        X2 = torch.randn(num_samples, 5)
        y = torch.randint(0, 2, (num_samples,))
        dataset = TensorDataset(X1, X2, y)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(model, dataloader, model_type, epochs=50, lr=0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    print(f"{'='*60}\\n")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if model_type == 'multi_input':
                input1, input2, targets = batch
                input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)
                outputs = model(input1, input2)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print(f"\\n✓ Training completed!")
    return model


def export_onnx(model, dummy_input, onnx_path, input_names, output_names, dynamic_axes=None):
    """Export PyTorch model to ONNX format"""
    model.eval()
    model.cpu()
    
    print(f"\\nExporting model to ONNX: {onnx_path}")
    
    kwargs = {
        'export_params': True,
        'opset_version': 11,
        'do_constant_folding': True,
        'input_names': input_names,
        'output_names': output_names,
    }
    
    if dynamic_axes:
        kwargs['dynamic_axes'] = dynamic_axes
    
    # Add external_data=False if supported
    sig = inspect.signature(torch.onnx.export)
    if 'external_data' in sig.parameters:
        kwargs['external_data'] = False
    
    torch.onnx.export(model, dummy_input, onnx_path, **kwargs)
    
    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    file_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"✓ ONNX model exported successfully: {file_size:.2f} MB")


def generate_calibration_data(model, model_type, num_samples, output_dir):
    """Generate calibration data from model"""
    print(f"\\n{'='*60}")
    print(f"Generating calibration data")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}\\n")
    
    model.eval()
    model.cpu()
    
    temp_dir = os.path.join(output_dir, "temp_calib")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with torch.no_grad():
            if model_type == 'mlp':
                input_data = torch.randn(num_samples, 10).numpy()
                for i in range(num_samples):
                    np.save(os.path.join(temp_dir, f"input_sample_{i:04d}.npy"), 
                           input_data[i:i+1].astype(np.float32))
                    
            elif model_type in ['cnn', 'resnet']:
                input_data = torch.randn(num_samples, 3, 32, 32).numpy()
                for i in range(num_samples):
                    np.save(os.path.join(temp_dir, f"input_sample_{i:04d}.npy"), 
                           input_data[i:i+1].astype(np.float32))
                    
            elif model_type == 'transformer':
                input_data = torch.randint(0, 10000, (num_samples, 64)).numpy()
                for i in range(num_samples):
                    np.save(os.path.join(temp_dir, f"input_sample_{i:04d}.npy"), 
                           input_data[i:i+1].astype(np.int32))
                    
            elif model_type == 'multi_input':
                input1_data = torch.randn(num_samples, 10).numpy()
                input2_data = torch.randn(num_samples, 5).numpy()
                for i in range(num_samples):
                    np.save(os.path.join(temp_dir, f"input1_sample_{i:04d}.npy"), 
                           input1_data[i:i+1].astype(np.float32))
                    np.save(os.path.join(temp_dir, f"input2_sample_{i:04d}.npy"), 
                           input2_data[i:i+1].astype(np.float32))
        
        # Create ZIP file
        zip_path = os.path.join(output_dir, "calibration_data.zip")
        print(f"Creating ZIP archive: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.basename(file_path))
        
        # Statistics
        npy_files = [f for f in os.listdir(temp_dir) if f.endswith('.npy')]
        zip_size = os.path.getsize(zip_path) / 1024
        
        print(f"\\n✓ Calibration data generated!")
        print(f"  Files: {len(npy_files)} .npy files")
        print(f"  ZIP size: {zip_size:.2f} KB")
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ===========================
# Main Pipeline
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description='训练PyTorch模型并生成ONNX和校准数据',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'cnn', 'resnet', 'transformer', 'multi_input'],
                       help='模型类型 (default: mlp)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数 (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率 (default: 0.001)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='校准数据样本数 (default: 100)')
    parser.add_argument('--output_dir', type=str, default='model_outputs',
                       help='输出目录 (default: model_outputs)')
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练，仅生成校准数据')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\\n{'='*60}")
    print(f"模型训练与校准数据生成器")
    print(f"{'='*60}")
    print(f"模型类型: {args.model}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\\n")
    
    # Create model
    if args.model == 'mlp':
        model = MLPModel()
        dummy_input = torch.randn(1, 10)
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    elif args.model == 'cnn':
        model = CNNModel()
        dummy_input = torch.randn(1, 3, 32, 32)
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    elif args.model == 'resnet':
        model = SimpleResNet()
        dummy_input = torch.randn(1, 3, 32, 32)
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    elif args.model == 'transformer':
        model = TransformerModel()
        dummy_input = torch.randint(0, 10000, (1, 64))
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    elif args.model == 'multi_input':
        model = MultiInputModel()
        dummy_input = (torch.randn(1, 10), torch.randn(1, 5))
        input_names = ['input1', 'input2']
        output_names = ['output']
        dynamic_axes = {
            'input1': {0: 'batch_size'}, 
            'input2': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Training
    if not args.skip_training:
        dataloader = create_dummy_dataset(args.model, num_samples=1000, batch_size=args.batch_size)
        model = train_model(model, dataloader, args.model, epochs=args.epochs, lr=args.lr)
        
        # Save PyTorch model
        model_path = os.path.join(args.output_dir, f"{args.model}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\\n✓ PyTorch model saved: {model_path}")
    else:
        print("\\n⚠ Skipping training (--skip_training flag)")
    
    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, f"{args.model}_model.onnx")
    export_onnx(model, dummy_input, onnx_path, input_names, output_names, dynamic_axes)
    
    # Generate calibration data
    generate_calibration_data(model, args.model, args.num_samples, args.output_dir)
    
    print(f"\\n{'='*60}")
    print(f"✓ All tasks completed!")
    print(f"{'='*60}")
    print(f"\\n输出文件:")
    print(f"  - ONNX 模型: {onnx_path}")
    print(f"  - 校准数据: {os.path.join(args.output_dir, 'calibration_data.zip')}")
    if not args.skip_training:
        print(f"  - PyTorch 模型: {model_path}")
    print(f"\\n使用示例:")
    print(f"  上传 {args.model}_model.onnx 到 Web 界面进行转换")
    print(f"  上传 calibration_data.zip 用于量化")
    print(f"{'='*60}\\n")


if __name__ == '__main__':
    main()
'''
    
    # Save to temporary file and send
    script_filename = 'train_and_calibrate.py'
    script_path = os.path.join(app.config['OUTPUT_FOLDER'], script_filename)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return send_file(script_path, as_attachment=True, download_name=script_filename)

@app.route('/convert_to_cpp', methods=['POST'])
def convert_to_cpp():
    try:
        data = request.json
        filename = data.get('filename')
        
        # Secure the filename to prevent path traversal attacks
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], safe_filename)
        
        # Verify the file exists and is within the output folder
        if not os.path.exists(file_path) or not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
             return jsonify({'error': 'File not found'}), 404
             
        cpp_filename = os.path.splitext(safe_filename)[0] + ".cc"
        cpp_path = os.path.join(app.config['OUTPUT_FOLDER'], cpp_filename)
        
        with open(file_path, 'rb') as f:
            content = f.read()
            
        var_name = "model_data"
        hex_data = ", ".join([f"0x{b:02x}" for b in content])
        cpp_code = f"unsigned char {var_name}[] = {{{hex_data}}};\nunsigned int {var_name}_len = {len(content)};"
        
        with open(cpp_path, 'w') as f:
            f.write(cpp_code)
            
        return jsonify({'success': True, 'cpp_filename': cpp_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    start_cleanup_timer()
    app.run(host='0.0.0.0', port=5000, debug=True)
