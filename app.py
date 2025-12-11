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

# 导入转换器
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eiq-onnx2tflite'))
import onnx2tflite.src.converter.convert as convert

# 导入量化相关模块
from onnx2quant.qdq_quantization import QDQQuantizer, RandomDataCalibrationDataReader, InputSpec
from onnx2quant.quantization_config import QuantizationConfig
import numpy as np

# 导入推理相关模块
import onnxruntime as ort
import tensorflow as tf
from typing import Dict, List, Tuple, Any

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

ALLOWED_EXTENSIONS = {'onnx', 'zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """清理1分钟前的文件"""
    try:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=10)
        
        # 清理 uploads 文件夹
        upload_pattern = os.path.join(app.config['UPLOAD_FOLDER'], '*')
        for file_path in glob.glob(upload_pattern):
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        print(f"Deleted upload file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting upload file {file_path}: {e}")
        
        # 清理 outputs 文件夹
        output_pattern = os.path.join(app.config['OUTPUT_FOLDER'], '*')
        for file_path in glob.glob(output_pattern):
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        print(f"Deleted output file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting output file {file_path}: {e}")
                        
    except Exception as e:
        print(f"Error in cleanup_old_files: {e}")

def start_cleanup_timer():
    """启动定时清理线程"""
    def cleanup_loop():
        while True:
            time.sleep(600)  # 每10分钟检查一次
            cleanup_old_files()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("File cleanup timer started")

def get_model_info(onnx_path):
    """提取 ONNX 模型详细信息"""
    try:
        model = onnx.load(onnx_path)
        
        # 基本信息
        info = {
            'ir_version': model.ir_version,
            'producer_name': model.producer_name,
            'producer_version': model.producer_version,
            'domain': model.domain,
            'model_version': model.model_version,
            'doc_string': model.doc_string,
        }
        
        # Opset 版本
        opset_imports = []
        for opset in model.opset_import:
            opset_imports.append({
                'domain': opset.domain if opset.domain else 'ai.onnx',
                'version': opset.version
            })
        info['opset_imports'] = opset_imports
        
        # 图信息
        graph = model.graph
        info['graph_name'] = graph.name
        
        # 输入信息
        inputs = []
        for inp in graph.input:
            input_info = {
                'name': inp.name,
                'type': str(inp.type.tensor_type.elem_type),
                'shape': []
            }
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    input_info['shape'].append(dim.dim_param)
                else:
                    input_info['shape'].append(dim.dim_value)
            inputs.append(input_info)
        info['inputs'] = inputs
        
        # 输出信息
        outputs = []
        for out in graph.output:
            output_info = {
                'name': out.name,
                'type': str(out.type.tensor_type.elem_type),
                'shape': []
            }
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_param:
                    output_info['shape'].append(dim.dim_param)
                else:
                    output_info['shape'].append(dim.dim_value)
            outputs.append(output_info)
        info['outputs'] = outputs
        
        # 节点统计
        node_types = {}
        for node in graph.node:
            if node.op_type in node_types:
                node_types[node.op_type] += 1
            else:
                node_types[node.op_type] = 1
        
        info['total_nodes'] = len(graph.node)
        info['node_types'] = node_types
        info['total_initializers'] = len(graph.initializer)
        
        # 节点详细列表
        nodes = []
        for i, node in enumerate(graph.node[:50]):  # 只显示前50个节点
            nodes.append({
                'index': i,
                'op_type': node.op_type,
                'name': node.name if node.name else f"Node_{i}",
                'inputs': list(node.input),
                'outputs': list(node.output)
            })
        info['nodes'] = nodes
        if len(graph.node) > 50:
            info['nodes_truncated'] = True
            info['total_nodes_count'] = len(graph.node)
        
        return info
        
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        onnx_filename = f"{timestamp}_{filename}"
        onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], onnx_filename)
        
        file.save(onnx_path)
        
        # 获取模型信息
        model_info = get_model_info(onnx_path)
        
        return jsonify({
            'success': True,
            'filename': onnx_filename,
            'model_info': model_info
        })
    
    return jsonify({'error': 'Invalid file type. Only .onnx and .zip files are allowed'}), 400

@app.route('/upload_calibration_data', methods=['POST'])
def upload_calibration_data():
    """上传校准数据zip文件"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith('.zip'):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"calibration_{timestamp}_{filename}"
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
        
        file.save(zip_path)
        
        # 验证zip文件内容
        try:
            import zipfile
            npy_count = 0
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.npy'):
                        npy_count += 1
            
            if npy_count == 0:
                os.remove(zip_path)
                return jsonify({'error': 'ZIP文件中未找到.npy文件'}), 400
            
            return jsonify({
                'success': True,
                'filename': zip_filename,
                'npy_files_count': npy_count,
                'message': f'成功上传校准数据，包含 {npy_count} 个.npy文件'
            })
            
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return jsonify({'error': f'ZIP文件验证失败: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file type. Only .zip files are allowed for calibration data'}), 400

@app.route('/convert', methods=['POST'])
def convert_model():
    try:
        data = request.json
        onnx_filename = data.get('filename')
        options = data.get('options', {})
        
        if not onnx_filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], onnx_filename)
        
        if not os.path.exists(onnx_path):
            return jsonify({'error': 'ONNX file not found'}), 404
        
        # 生成输出文件名
        base_name = os.path.splitext(onnx_filename)[0]
        tflite_filename = f"{base_name}.tflite"
        tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], tflite_filename)
        
        # 获取转换前的模型信息
        start_time = datetime.now()
        original_model_info = get_model_info(onnx_path)
        
        # 转换模型
        print(f"Converting {onnx_path} to {tflite_path}")
        
        # 使用 convert_model 函数
        binary_tflite_model = convert.convert_model(onnx_path)
        
        # 保存 TFLite 模型
        with open(tflite_path, 'wb') as f:
            f.write(binary_tflite_model)
        
        conversion_time = datetime.now() - start_time
        
        # 获取文件大小
        onnx_size = os.path.getsize(onnx_path)
        tflite_size = os.path.getsize(tflite_path)
        compression_ratio = (1 - tflite_size/onnx_size) * 100
        
        # 生成转换报告
        conversion_report = {
            'conversion_time': f"{conversion_time.total_seconds():.2f} 秒",
            'original_format': 'ONNX',
            'target_format': 'TensorFlow Lite',
            'file_sizes': {
                'onnx_size': onnx_size,
                'onnx_size_mb': f"{onnx_size / 1024 / 1024:.2f} MB",
                'tflite_size': tflite_size,
                'tflite_size_mb': f"{tflite_size / 1024 / 1024:.2f} MB",
                'compression_ratio': f"{compression_ratio:.2f}%",
                'size_reduction': f"{(onnx_size - tflite_size) / 1024 / 1024:.2f} MB"
            },
            'model_structure': {
                'total_nodes': original_model_info.get('total_nodes', 'N/A'),
                'total_initializers': original_model_info.get('total_initializers', 'N/A'),
                'input_tensors': len(original_model_info.get('inputs', [])),
                'output_tensors': len(original_model_info.get('outputs', [])),
                'node_types': original_model_info.get('node_types', {})
            },
            'conversion_options': options,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'SUCCESS',
            'accuracy_comparison': None  # 将在前端触发精度对比后填充
        }
        
        return jsonify({
            'success': True,
            'tflite_filename': tflite_filename,
            'onnx_size': onnx_size,
            'tflite_size': tflite_size,
            'compression_ratio': f"{compression_ratio:.2f}%",
            'conversion_report': conversion_report
        })
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Conversion error: {error_msg}")
        print(trace)
        return jsonify({
            'error': error_msg,
            'traceback': trace
        }), 500

@app.route('/quantize', methods=['POST'])
def quantize_model():
    try:
        data = request.json
        onnx_filename = data.get('filename')
        selected_layers = data.get('selected_layers', [])
        calibration_data_filename = data.get('calibration_data_filename', None)
        
        if not onnx_filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], onnx_filename)
        
        if not os.path.exists(onnx_path):
            return jsonify({'error': 'ONNX file not found'}), 404
        
        # 生成量化ONNX模型文件名和TFLite文件名
        base_name = os.path.splitext(onnx_filename)[0]
        quantized_onnx_filename = f"{base_name}_quantized.onnx"
        quantized_onnx_path = os.path.join(app.config['OUTPUT_FOLDER'], quantized_onnx_filename)
        quantized_tflite_filename = f"{base_name}_quantized.tflite"
        quantized_tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], quantized_tflite_filename)
        
        start_time = datetime.now()
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 创建校准数据读取器
        validation_report = None
        if calibration_data_filename:
            calibration_data_path = os.path.join(app.config['UPLOAD_FOLDER'], calibration_data_filename)
            if os.path.exists(calibration_data_path):
                calibration_data_reader, validation_report = create_real_data_calibration_reader(onnx_model, calibration_data_path)
                print(f"Using real calibration data from: {calibration_data_filename}")
            else:
                print(f"Calibration data file not found, using random data")
                calibration_data_reader = create_calibration_data_reader(onnx_model)
        else:
            print("No calibration data provided, using random data")
            calibration_data_reader = create_calibration_data_reader(onnx_model)
        
        quantization_config = QuantizationConfig(calibration_data_reader)
        
        # 创建QDQ量化器，只量化选定的层类型
        quantizer = QDQQuantizer(op_types_to_quantize=selected_layers)
        
        print(f"Quantizing model with layers: {selected_layers}")
        
        # 量化模型
        quantized_onnx_model = quantizer.quantize_model(onnx_model, quantization_config)
        
        # 保存量化后的ONNX模型
        onnx.save(quantized_onnx_model, quantized_onnx_path)
        
        # 转换量化后的ONNX模型到TFLite
        quantized_tflite_model = convert.convert_model(quantized_onnx_path)
        
        # 保存量化后的TFLite模型
        with open(quantized_tflite_path, 'wb') as f:
            f.write(quantized_tflite_model)
        
        quantization_time = datetime.now() - start_time
        
        # 获取文件大小信息
        onnx_size = os.path.getsize(onnx_path)
        quantized_onnx_size = os.path.getsize(quantized_onnx_path)
        quantized_tflite_size = os.path.getsize(quantized_tflite_path)
        
        # 计算压缩比
        additional_compression = (1 - quantized_tflite_size/onnx_size) * 100
        
        # 构建量化报告
        quantization_report = {
            'quantization_time': f"{quantization_time.total_seconds():.2f} 秒",
            'quantization_method': 'QDQ (Quantization-Dequantization)',
            'quantized_layer_types': selected_layers,
            'file_sizes': {
                'original_onnx_size': onnx_size,
                'original_onnx_size_mb': f"{onnx_size / 1024 / 1024:.2f} MB",
                'quantized_onnx_size': quantized_onnx_size,
                'quantized_onnx_size_mb': f"{quantized_onnx_size / 1024 / 1024:.2f} MB",
                'quantized_tflite_size': quantized_tflite_size,
                'quantized_tflite_size_mb': f"{quantized_tflite_size / 1024 / 1024:.2f} MB",
                'additional_compression_ratio': f"{additional_compression:.2f}%"
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'SUCCESS',
            'accuracy_comparison': None,  # 将在前端触发精度对比后填充
        }
        
        return jsonify({
            'success': True,
            'quantized_onnx_filename': quantized_onnx_filename,
            'quantized_tflite_filename': quantized_tflite_filename,
            'original_onnx_size': onnx_size,
            'quantized_onnx_size': quantized_onnx_size,
            'quantized_tflite_size': quantized_tflite_size,
            'additional_compression_ratio': f"{additional_compression:.2f}%",
            'quantization_time': f"{quantization_time.total_seconds():.2f} 秒",
            'quantized_layers': selected_layers,
            'quantization_report': quantization_report,
            'validation_report': validation_report
        })
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Quantization error: {error_msg}")
        print(trace)
        return jsonify({
            'error': error_msg,
            'traceback': trace
        }), 500


def create_calibration_data_reader(onnx_model):
    """创建校准数据读取器"""
    try:
        # 获取模型输入信息
        inputs = {}
        initializer_names = [init.name for init in onnx_model.graph.initializer]
        
        for input_info in onnx_model.graph.input:
            if input_info.name in initializer_names:
                continue  # 跳过初始化器
                
            # 解析输入维度
            dims = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    dims.append(dim.dim_value)
                elif dim.dim_param:
                    # 动态维度，使用常见的默认值
                    if 'batch' in dim.dim_param.lower():
                        dims.append(1)  # 批大小默认为1
                    else:
                        dims.append(224)  # 其他动态维度默认为224
                else:
                    dims.append(1)  # 未知维度默认为1
            
            # 确定数据类型
            elem_type = input_info.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT:
                numpy_type = np.float32
            elif elem_type == onnx.TensorProto.INT64:
                numpy_type = np.int64
            else:
                numpy_type = np.float32  # 默认类型
            
            inputs[input_info.name] = InputSpec(dims, numpy_type)
        
        return RandomDataCalibrationDataReader(inputs, num_samples=5)
    
    except Exception as e:
        print(f"Error creating calibration data reader: {e}")
        # 返回一个默认的校准数据读取器
        default_inputs = {"input": InputSpec([1, 3, 224, 224], np.float32)}
        return RandomDataCalibrationDataReader(default_inputs, num_samples=5)


def create_real_data_calibration_reader(onnx_model, calibration_data_path):
    """创建基于真实数据的校准数据读取器，返回读取器和验证报告"""
    import zipfile
    import tempfile
    import shutil
    
    # 初始化验证报告
    validation_report = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'validation_errors': [],
        'file_details': [],
        'expected_specs': {},
        'loaded_samples': 0,
        'status': 'processing'
    }
    
    try:
        # 获取模型输入信息
        input_info_dict = {}
        initializer_names = [init.name for init in onnx_model.graph.initializer]
        
        for input_info in onnx_model.graph.input:
            if input_info.name in initializer_names:
                continue
            
            # 解析输入维度和类型，确保第一维是batch=1
            dims = []
            for i, dim in enumerate(input_info.type.tensor_type.shape.dim):
                if i == 0:  # 第一维设置为batch=1
                    dims.append(1)
                elif dim.dim_value > 0:
                    dims.append(dim.dim_value)
                elif dim.dim_param:
                    if 'batch' in dim.dim_param.lower():
                        dims.append(1)
                    else:
                        dims.append(224)  # 默认尺寸
                else:
                    dims.append(1)
            
            elem_type = input_info.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT:
                numpy_type = np.float32
                dtype_str = 'float32'
            elif elem_type == onnx.TensorProto.INT64:
                numpy_type = np.int64
                dtype_str = 'int64'
            else:
                numpy_type = np.float32
                dtype_str = 'float32'
                
            input_info_dict[input_info.name] = {
                'shape': dims,
                'dtype': numpy_type,  # 用于实际数据处理
                'dtype_str': dtype_str  # 用于JSON序列化
            }
        
        # 为JSON序列化创建可序列化的expected_specs
        serializable_specs = {}
        for name, specs in input_info_dict.items():
            serializable_specs[name] = {
                'shape': specs['shape'],
                'dtype': specs['dtype_str']
            }
        validation_report['expected_specs'] = serializable_specs
        print(f"Expected input specifications (batch=1): {serializable_specs}")
        
        # 解压zip文件并加载npy数据
        calibration_data = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解压zip文件
            with zipfile.ZipFile(calibration_data_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 查找所有npy文件
            npy_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.npy'):
                        npy_files.append(os.path.join(root, file))
            
            validation_report['total_files'] = len(npy_files)
            print(f"Found {len(npy_files)} npy files in calibration data")
            
            # 加载并验证每个npy文件
            for npy_file in npy_files:
                file_detail = {
                    'filename': os.path.basename(npy_file),
                    'status': 'failed',
                    'actual_shape': None,
                    'actual_dtype': None,
                    'actual_dtype': None,
                    'matched_input': None,
                    'error': None
                }
                
                try:
                    data = np.load(npy_file)
                    file_detail['actual_shape'] = data.shape
                    file_detail['actual_dtype'] = str(data.dtype)
                    
                    print(f"Loaded {os.path.basename(npy_file)}: shape={data.shape}, dtype={data.dtype}")
                    
                    # 记录实际的数据信息
                    file_detail['actual_shape'] = list(data.shape)
                    file_detail['actual_dtype'] = str(data.dtype)
                    
                    # 验证数据形状是否符合batch=1的要求
                    matched_input = None
                    shape_valid = False
                    
                    # 如果只有一个输入，直接匹配
                    if len(input_info_dict) == 1:
                        input_name = list(input_info_dict.keys())[0]
                        expected_shape = input_info_dict[input_name]['shape']
                        expected_dtype = input_info_dict[input_name]['dtype']
                        
                        # 检查是否已经是batch=1的形状
                        if data.shape == tuple(expected_shape):
                            matched_input = input_name
                            shape_valid = True
                        # 检查是否缺少batch维度
                        elif data.shape == tuple(expected_shape[1:]) and expected_shape[0] == 1:
                            matched_input = input_name
                            shape_valid = True
                            # 添加batch维度
                            data = np.expand_dims(data, axis=0)
                            print(f"  Added batch dimension: {data.shape}")
                        # 检查是否可以reshape
                        elif data.size == np.prod(expected_shape):
                            matched_input = input_name
                            shape_valid = True
                            data = data.reshape(expected_shape)
                            print(f"  Reshaped to: {data.shape}")
                    else:
                        # 多输入情况，尝试根据文件名或形状匹配
                        for input_name, specs in input_info_dict.items():
                            expected_shape = specs['shape']
                            
                            # 文件名匹配
                            if input_name.lower() in os.path.basename(npy_file).lower():
                                # 验证形状
                                if data.shape == tuple(expected_shape):
                                    matched_input = input_name
                                    shape_valid = True
                                    break
                                elif data.shape == tuple(expected_shape[1:]) and expected_shape[0] == 1:
                                    matched_input = input_name
                                    shape_valid = True
                                    data = np.expand_dims(data, axis=0)
                                    break
                    
                    if not shape_valid or not matched_input:
                        error_msg = f"Shape mismatch: got {data.shape}, expected one of {[tuple(specs['shape']) for specs in input_info_dict.values()]}"
                        file_detail['error'] = error_msg
                        validation_report['invalid_files'] += 1
                        validation_report['validation_errors'].append(f"{os.path.basename(npy_file)}: {error_msg}")
                        print(f"  ❌ {error_msg}")
                    else:
                        # 确保数据类型正确
                        expected_dtype = input_info_dict[matched_input]['dtype']
                        if data.dtype != expected_dtype:
                            data = data.astype(expected_dtype)
                            print(f"  Converted dtype to: {expected_dtype}")
                        
                        # 最终验证：确保是batch=1
                        if data.shape[0] != 1:
                            error_msg = f"Batch size must be 1, got {data.shape[0]}"
                            file_detail['error'] = error_msg
                            validation_report['invalid_files'] += 1
                            validation_report['validation_errors'].append(f"{os.path.basename(npy_file)}: {error_msg}")
                            print(f"  ❌ {error_msg}")
                        else:
                            sample = {matched_input: data}
                            calibration_data.append(sample)
                            file_detail['status'] = 'success'
                            file_detail['matched_input'] = matched_input
                            validation_report['valid_files'] += 1
                            print(f"  ✓ Valid sample for input '{matched_input}': {data.shape}")
                        
                except Exception as e:
                    error_msg = f"Error loading file: {str(e)}"
                    file_detail['error'] = error_msg
                    validation_report['invalid_files'] += 1
                    validation_report['validation_errors'].append(f"{os.path.basename(npy_file)}: {error_msg}")
                    print(f"  ❌ Error loading {os.path.basename(npy_file)}: {e}")
                
                validation_report['file_details'].append(file_detail)
        
        validation_report['loaded_samples'] = len(calibration_data)
        validation_report['status'] = 'completed'
        
        # 生成验证报告摘要
        print(f"\n=== Calibration Data Validation Report ===")
        print(f"Total files found: {validation_report['total_files']}")
        print(f"Valid files: {validation_report['valid_files']}")
        print(f"Invalid files: {validation_report['invalid_files']}")
        print(f"Successfully loaded samples: {validation_report['loaded_samples']}")
        
        if validation_report['validation_errors']:
            print(f"\nValidation Errors:")
            for error in validation_report['validation_errors'][:10]:  # 显示前10个错误
                print(f"  - {error}")
            if len(validation_report['validation_errors']) > 10:
                print(f"  ... and {len(validation_report['validation_errors']) - 10} more errors")
        
        if not calibration_data:
            print("\nNo valid calibration data found, using random data")
            return create_calibration_data_reader(onnx_model), validation_report
        
        print(f"\n✓ Successfully loaded {len(calibration_data)} calibration samples")
        
        # 创建自定义校准数据读取器
        class RealDataCalibrationReader:
            def __init__(self, calibration_data, validation_report):
                self.calibration_data = calibration_data
                self.validation_report = validation_report
                self.iter_index = 0
            
            def get_next(self):
                if self.iter_index < len(self.calibration_data):
                    data = self.calibration_data[self.iter_index]
                    self.iter_index += 1
                    return data
                else:
                    return None
            
            def get_validation_report(self):
                return self.validation_report
        
        return RealDataCalibrationReader(calibration_data, validation_report), validation_report
        
    except Exception as e:
        error_msg = f"Error creating real data calibration reader: {e}"
        print(error_msg)
        validation_report['validation_errors'].append(error_msg)
        validation_report['status'] = 'error'
        print(f"Falling back to random data calibration reader")
        return create_calibration_data_reader(onnx_model), validation_report


def generate_test_data(input_specs: Dict[str, Tuple], num_samples: int = 100) -> List[Dict[str, np.ndarray]]:
    """生成测试数据"""
    test_data = []
    np.random.seed(42)  # 设置随机种子确保可重现性
    
    for i in range(num_samples):
        sample = {}
        for input_name, (shape, dtype) in input_specs.items():
            if dtype in [np.float32, np.float64]:
                # 生成正态分布的浮点数据，范围在-1到1之间
                sample[input_name] = np.random.normal(0, 0.5, shape).astype(dtype)
            elif dtype in [np.int32, np.int64]:
                # 生成整数数据
                sample[input_name] = np.random.randint(0, 100, shape).astype(dtype)
            else:
                # 默认生成浮点数据
                sample[input_name] = np.random.normal(0, 0.5, shape).astype(np.float32)
        test_data.append(sample)
    
    return test_data


def run_onnx_inference(onnx_path: str, test_data: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """运行ONNX模型推理"""
    try:
        # 创建ONNX Runtime会话
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        results = []
        for sample in test_data:
            try:
                # 运行推理
                outputs = session.run(None, sample)
                # 将输出转换为字典格式
                output_names = [output.name for output in session.get_outputs()]
                result = {name: output for name, output in zip(output_names, outputs)}
                results.append(result)
            except Exception as e:
                print(f"ONNX inference error for sample: {e}")
                results.append({})
        
        return results
    except Exception as e:
        print(f"ONNX model loading error: {e}")
        return []


def run_tflite_inference(tflite_path: str, test_data: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """运行TFLite模型推理"""
    try:
        # 加载TFLite模型
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # 获取输入和输出详情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        results = []
        for sample in test_data:
            try:
                # 设置输入数据
                for input_detail in input_details:
                    input_name = input_detail['name']
                    input_index = input_detail['index']
                    
                    # 查找对应的输入数据
                    input_data = None
                    for key, value in sample.items():
                        if key == input_name or key in input_name or input_name in key:
                            input_data = value
                            break
                    
                    if input_data is not None:
                        # 确保数据类型和形状匹配
                        if input_data.dtype != input_detail['dtype']:
                            input_data = input_data.astype(input_detail['dtype'])
                        if input_data.shape != tuple(input_detail['shape']):
                            # 如果形状不匹配，尝试reshape
                            input_data = np.reshape(input_data, input_detail['shape'])
                        
                        interpreter.set_tensor(input_index, input_data)
                
                # 运行推理
                interpreter.invoke()
                
                # 获取输出
                result = {}
                for output_detail in output_details:
                    output_data = interpreter.get_tensor(output_detail['index'])
                    result[output_detail['name']] = output_data.copy()
                
                results.append(result)
            except Exception as e:
                print(f"TFLite inference error for sample: {e}")
                results.append({})
        
        return results
    except Exception as e:
        print(f"TFLite model loading error: {e}")
        return []


def calculate_accuracy_metrics(onnx_results: List[Dict], tflite_results: List[Dict], quantized_tflite_results: List[Dict] = None) -> Dict:
    """计算精度指标"""
    metrics = {
        'total_samples': len(onnx_results),
        'successful_samples': 0,
        'onnx_vs_tflite': {},
        'onnx_vs_quantized_tflite': {}
    }
    
    if not onnx_results or not tflite_results:
        return metrics
    
    # ONNX vs TFLite 比较
    onnx_tflite_errors = []
    successful_comparisons = 0
    
    for i, (onnx_result, tflite_result) in enumerate(zip(onnx_results, tflite_results)):
        if not onnx_result or not tflite_result:
            continue
        
        try:
            # 假设两个模型都只有一个输出，取第一个输出进行比较
            onnx_output = list(onnx_result.values())[0] if onnx_result else None
            tflite_output = list(tflite_result.values())[0] if tflite_result else None
            
            if onnx_output is not None and tflite_output is not None:
                # 确保形状一致
                if onnx_output.shape != tflite_output.shape:
                    # 尝试reshape
                    min_size = min(onnx_output.size, tflite_output.size)
                    onnx_output = onnx_output.flatten()[:min_size]
                    tflite_output = tflite_output.flatten()[:min_size]
                
                # 计算各种误差指标
                mse = np.mean((onnx_output - tflite_output) ** 2)
                mae = np.mean(np.abs(onnx_output - tflite_output))
                max_error = np.max(np.abs(onnx_output - tflite_output))
                
                # 计算相对误差（避免除零）
                rel_error = np.mean(np.abs(onnx_output - tflite_output) / (np.abs(onnx_output) + 1e-8))
                
                onnx_tflite_errors.append({
                    'mse': float(mse),
                    'mae': float(mae),
                    'max_error': float(max_error),
                    'relative_error': float(rel_error)
                })
                
                successful_comparisons += 1
        except Exception as e:
            print(f"Error comparing sample {i}: {e}")
            continue
    
    metrics['successful_samples'] = successful_comparisons
    
    if onnx_tflite_errors:
        # 计算统计指标
        metrics['onnx_vs_tflite'] = {
            'mean_mse': float(np.mean([e['mse'] for e in onnx_tflite_errors])),
            'mean_mae': float(np.mean([e['mae'] for e in onnx_tflite_errors])),
            'mean_max_error': float(np.mean([e['max_error'] for e in onnx_tflite_errors])),
            'mean_relative_error': float(np.mean([e['relative_error'] for e in onnx_tflite_errors])),
            'max_mse': float(np.max([e['mse'] for e in onnx_tflite_errors])),
            'max_mae': float(np.max([e['mae'] for e in onnx_tflite_errors])),
            'max_max_error': float(np.max([e['max_error'] for e in onnx_tflite_errors])),
            'max_relative_error': float(np.max([e['relative_error'] for e in onnx_tflite_errors]))
        }
    
    # ONNX vs 量化TFLite 比较（如果提供了量化模型结果）
    if quantized_tflite_results:
        onnx_quantized_errors = []
        
        for i, (onnx_result, quantized_result) in enumerate(zip(onnx_results, quantized_tflite_results)):
            if not onnx_result or not quantized_result:
                continue
            
            try:
                onnx_output = list(onnx_result.values())[0] if onnx_result else None
                quantized_output = list(quantized_result.values())[0] if quantized_result else None
                
                if onnx_output is not None and quantized_output is not None:
                    # 确保形状一致
                    if onnx_output.shape != quantized_output.shape:
                        min_size = min(onnx_output.size, quantized_output.size)
                        onnx_output = onnx_output.flatten()[:min_size]
                        quantized_output = quantized_output.flatten()[:min_size]
                    
                    # 计算误差指标
                    mse = np.mean((onnx_output - quantized_output) ** 2)
                    mae = np.mean(np.abs(onnx_output - quantized_output))
                    max_error = np.max(np.abs(onnx_output - quantized_output))
                    rel_error = np.mean(np.abs(onnx_output - quantized_output) / (np.abs(onnx_output) + 1e-8))
                    
                    onnx_quantized_errors.append({
                        'mse': float(mse),
                        'mae': float(mae),
                        'max_error': float(max_error),
                        'relative_error': float(rel_error)
                    })
            except Exception as e:
                print(f"Error comparing quantized sample {i}: {e}")
                continue
        
        if onnx_quantized_errors:
            metrics['onnx_vs_quantized_tflite'] = {
                'mean_mse': float(np.mean([e['mse'] for e in onnx_quantized_errors])),
                'mean_mae': float(np.mean([e['mae'] for e in onnx_quantized_errors])),
                'mean_max_error': float(np.mean([e['max_error'] for e in onnx_quantized_errors])),
                'mean_relative_error': float(np.mean([e['relative_error'] for e in onnx_quantized_errors])),
                'max_mse': float(np.max([e['mse'] for e in onnx_quantized_errors])),
                'max_mae': float(np.max([e['mae'] for e in onnx_quantized_errors])),
                'max_max_error': float(np.max([e['max_error'] for e in onnx_quantized_errors])),
                'max_relative_error': float(np.max([e['relative_error'] for e in onnx_quantized_errors]))
            }
    
    return metrics


def get_model_input_specs(onnx_path: str) -> Dict[str, Tuple]:
    """获取模型输入规格"""
    try:
        model = onnx.load(onnx_path)
        input_specs = {}
        initializer_names = [init.name for init in model.graph.initializer]
        
        for input_info in model.graph.input:
            if input_info.name in initializer_names:
                continue
            
            # 解析输入维度
            dims = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    dims.append(dim.dim_value)
                elif dim.dim_param:
                    # 动态维度，使用默认值
                    if 'batch' in dim.dim_param.lower():
                        dims.append(1)
                    else:
                        dims.append(224)
                else:
                    dims.append(1)
            
            # 确定数据类型
            elem_type = input_info.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT:
                numpy_type = np.float32
            elif elem_type == onnx.TensorProto.INT64:
                numpy_type = np.int64
            else:
                numpy_type = np.float32
            
            input_specs[input_info.name] = (tuple(dims), numpy_type)
        
        return input_specs
    
    except Exception as e:
        print(f"Error getting model input specs: {e}")
        # 返回默认规格
        return {"input": ((1, 3, 224, 224), np.float32)}


@app.route('/accuracy_comparison', methods=['POST'])
def accuracy_comparison():
    """执行精度对比分析"""
    try:
        data = request.json
        onnx_filename = data.get('filename')
        tflite_filename = data.get('tflite_filename')
        quantized_tflite_filename = data.get('quantized_tflite_filename')
        num_samples = data.get('num_samples', 100)
        
        if not onnx_filename:
            return jsonify({'error': 'No ONNX filename provided'}), 400
        
        if not tflite_filename:
            return jsonify({'error': 'No TFLite filename provided'}), 400
        
        # 文件路径
        onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], onnx_filename)
        tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], tflite_filename)
        
        if not os.path.exists(onnx_path):
            return jsonify({'error': 'ONNX file not found'}), 404
        
        if not os.path.exists(tflite_path):
            return jsonify({'error': 'TFLite file not found'}), 404
        
        start_time = datetime.now()
        
        # 获取模型输入规格
        input_specs = get_model_input_specs(onnx_path)
        
        # 生成测试数据
        print(f"Generating {num_samples} test samples...")
        test_data = generate_test_data(input_specs, num_samples)
        
        # 运行ONNX推理
        print("Running ONNX inference...")
        onnx_results = run_onnx_inference(onnx_path, test_data)
        
        # 运行TFLite推理
        print("Running TFLite inference...")
        tflite_results = run_tflite_inference(tflite_path, test_data)
        
        # 运行量化TFLite推理（如果提供了量化模型）
        quantized_tflite_results = None
        if quantized_tflite_filename:
            quantized_tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], quantized_tflite_filename)
            if os.path.exists(quantized_tflite_path):
                print("Running quantized TFLite inference...")
                quantized_tflite_results = run_tflite_inference(quantized_tflite_path, test_data)
        
        # 计算精度指标
        print("Calculating accuracy metrics...")
        accuracy_metrics = calculate_accuracy_metrics(onnx_results, tflite_results, quantized_tflite_results)
        
        comparison_time = datetime.now() - start_time
        
        # 构建完整的精度对比报告
        accuracy_report = {
            'test_parameters': {
                'num_samples': num_samples,
                'input_specs': {name: {'shape': list(shape), 'dtype': str(dtype)} 
                              for name, (shape, dtype) in input_specs.items()},
                'comparison_time': f"{comparison_time.total_seconds():.2f} 秒"
            },
            'inference_results': {
                'onnx_successful_inferences': len([r for r in onnx_results if r]),
                'tflite_successful_inferences': len([r for r in tflite_results if r]),
                'quantized_tflite_successful_inferences': len([r for r in quantized_tflite_results if r]) if quantized_tflite_results else 0
            },
            'accuracy_metrics': accuracy_metrics,
            'summary': {}
        }
        
        # 添加汇总信息
        if accuracy_metrics.get('onnx_vs_tflite'):
            accuracy_report['summary']['onnx_vs_tflite'] = {
                'average_relative_error': f"{accuracy_metrics['onnx_vs_tflite']['mean_relative_error']:.6f}",
                'max_relative_error': f"{accuracy_metrics['onnx_vs_tflite']['max_relative_error']:.6f}",
                'average_mae': f"{accuracy_metrics['onnx_vs_tflite']['mean_mae']:.6f}",
                'max_mae': f"{accuracy_metrics['onnx_vs_tflite']['max_mae']:.6f}"
            }
        
        if accuracy_metrics.get('onnx_vs_quantized_tflite'):
            accuracy_report['summary']['onnx_vs_quantized_tflite'] = {
                'average_relative_error': f"{accuracy_metrics['onnx_vs_quantized_tflite']['mean_relative_error']:.6f}",
                'max_relative_error': f"{accuracy_metrics['onnx_vs_quantized_tflite']['max_relative_error']:.6f}",
                'average_mae': f"{accuracy_metrics['onnx_vs_quantized_tflite']['mean_mae']:.6f}",
                'max_mae': f"{accuracy_metrics['onnx_vs_quantized_tflite']['max_mae']:.6f}"
            }
        
        return jsonify({
            'success': True,
            'accuracy_report': accuracy_report
        })
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Accuracy comparison error: {error_msg}")
        print(trace)
        return jsonify({
            'error': error_msg,
            'traceback': trace
        }), 500


@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/convert_to_cpp', methods=['POST'])
def convert_to_cpp():
    """将TFLite模型转换为C++文件"""
    try:
        data = request.json
        tflite_filename = data.get('tflite_filename')
        
        if not tflite_filename:
            return jsonify({'error': 'No TFLite filename provided'}), 400
        
        tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], tflite_filename)
        
        if not os.path.exists(tflite_path):
            return jsonify({'error': 'TFLite file not found'}), 404
        
        # 检查是否有xxd命令（Windows可能需要安装Git Bash或WSL）
        xxd_available = False
        try:
            # 首先尝试直接调用xxd
            result = subprocess.run(['xxd', '-v'], capture_output=True, text=True)
            xxd_available = True
        except FileNotFoundError:
            # 如果xxd不可用，尝试使用Git Bash中的xxd
            try:
                result = subprocess.run(['C:\\Program Files\\Git\\usr\\bin\\xxd.exe', '-v'], 
                                       capture_output=True, text=True)
                xxd_available = True
            except FileNotFoundError:
                pass
        
        if not xxd_available:
            # 如果没有xxd，使用Python实现相同功能
            return convert_to_cpp_python(tflite_path, tflite_filename)
        
        # 生成C++文件名
        base_name = os.path.splitext(tflite_filename)[0]
        cpp_filename = f"{base_name}.cc"
        cpp_path = os.path.join(app.config['OUTPUT_FOLDER'], cpp_filename)
        
        # 使用xxd命令转换
        try:
            # 尝试使用系统的xxd
            result = subprocess.run(['xxd', '-i', tflite_path], 
                                   capture_output=True, text=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # 尝试使用Git Bash的xxd
            try:
                result = subprocess.run(['C:\\Program Files\\Git\\usr\\bin\\xxd.exe', '-i', tflite_path], 
                                       capture_output=True, text=True, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                return convert_to_cpp_python(tflite_path, tflite_filename)
        
        # 保存C++文件
        with open(cpp_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        
        # 获取文件大小
        tflite_size = os.path.getsize(tflite_path)
        cpp_size = os.path.getsize(cpp_path)
        
        return jsonify({
            'success': True,
            'cpp_filename': cpp_filename,
            'tflite_size': tflite_size,
            'cpp_size': cpp_size,
            'message': 'TFLite模型已成功转换为C++文件'
        })
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"TFLite to C++ conversion error: {error_msg}")
        print(trace)
        return jsonify({
            'error': error_msg,
            'traceback': trace
        }), 500

def convert_to_cpp_python(tflite_path, tflite_filename):
    """使用Python实现xxd -i的功能"""
    try:
        # 生成C++文件名
        base_name = os.path.splitext(tflite_filename)[0]
        cpp_filename = f"{base_name}.cc"
        cpp_path = os.path.join(app.config['OUTPUT_FOLDER'], cpp_filename)
        
        # 读取TFLite文件
        with open(tflite_path, 'rb') as f:
            data = f.read()
        
        # 生成变量名（替换非字母数字字符为下划线）
        var_name = os.path.splitext(os.path.basename(tflite_filename))[0]
        var_name = ''.join(c if c.isalnum() else '_' for c in var_name)
        
        # 生成C++代码
        cpp_content = []
        cpp_content.append(f"unsigned char {var_name}[] = {{")
        
        # 每行16个字节
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_values = [f"0x{b:02x}" for b in chunk]
            line = "  " + ", ".join(hex_values)
            if i + 16 < len(data):
                line += ","
            cpp_content.append(line)
        
        cpp_content.append("};")
        cpp_content.append(f"unsigned int {var_name}_len = {len(data)};")
        
        # 保存C++文件
        with open(cpp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cpp_content))
        
        # 获取文件大小
        tflite_size = os.path.getsize(tflite_path)
        cpp_size = os.path.getsize(cpp_path)
        
        return jsonify({
            'success': True,
            'cpp_filename': cpp_filename,
            'tflite_size': tflite_size,
            'cpp_size': cpp_size,
            'message': 'TFLite模型已成功转换为C++文件（使用Python实现）'
        })
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Python TFLite to C++ conversion error: {error_msg}")
        print(trace)
        return jsonify({
            'error': error_msg,
            'traceback': trace
        }), 500

@app.route('/visualize/<filename>')
def visualize_model(filename):
    """生成模型可视化"""
    try:
        onnx_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(onnx_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # 使用 netron 生成可视化数据
        # 这里返回一个简化的图结构
        model = onnx.load(onnx_path)
        graph = model.graph
        
        nodes_data = []
        edges_data = []
        
        for i, node in enumerate(graph.node):
            nodes_data.append({
                'id': f"node_{i}",
                'label': f"{node.op_type}\n{node.name or f'Node_{i}'}",
                'op_type': node.op_type
            })
            
            # 创建边
            for inp in node.input:
                edges_data.append({
                    'from': inp,
                    'to': f"node_{i}"
                })
        
        return jsonify({
            'nodes': nodes_data[:100],  # 限制节点数量
            'edges': edges_data[:200]   # 限制边数量
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # 启动文件清理定时器
    start_cleanup_timer()
    
    app.run(host='0.0.0.0', port=5000, debug=True)