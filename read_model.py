import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import json

class DrowsinessModelReader:
    def __init__(self, model_path):
        """
        Khởi tạo class để đọc model .h5
        
        Args:
            model_path: Đường dẫn đến file .h5
        """
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Tải model từ file .h5"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Đã tải model thành công từ: {self.model_path}")
            return self.model
        except Exception as e:
            print(f"✗ Lỗi khi tải model: {e}")
            return None
    
    def get_model_summary(self):
        """Hiển thị thông tin tổng quan về model"""
        if self.model is None:
            print("Vui lòng load model trước!")
            return
        
        print("\n" + "="*60)
        print("THÔNG TIN MODEL")
        print("="*60)
        self.model.summary()
        
    def get_model_info(self):
        """Lấy thông tin chi tiết về model"""
        if self.model is None:
            print("Vui lòng load model trước!")
            return None
        
        info = {
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "total_layers": len(self.model.layers),
            "trainable_params": self.model.count_params(),
            "optimizer": self.model.optimizer.get_config() if self.model.optimizer else None,
            "loss": self.model.loss if hasattr(self.model, 'loss') else None
        }
        
        print("\n" + "="*60)
        print("CHI TIẾT MODEL")
        print("="*60)
        for key, value in info.items():
            print(f"{key}: {value}")
        
        return info
    
    def get_layer_info(self):
        """Hiển thị thông tin chi tiết về từng layer"""
        if self.model is None:
            print("Vui lòng load model trước!")
            return
        
        print("\n" + "="*60)
        print("THÔNG TIN CÁC LAYER")
        print("="*60)
        
        for i, layer in enumerate(self.model.layers):
            print(f"\nLayer {i+1}: {layer.name}")
            print(f"  - Type: {layer.__class__.__name__}")
            
            # Lấy output shape an toàn
            try:
                output_shape = layer.output.shape if hasattr(layer, 'output') else "N/A"
                print(f"  - Output shape: {output_shape}")
            except:
                print(f"  - Output shape: N/A")
            
            print(f"  - Params: {layer.count_params()}")
            
            # Thông tin cấu hình
            config = layer.get_config()
            if 'activation' in config:
                print(f"  - Activation: {config['activation']}")
            if 'filters' in config:
                print(f"  - Filters: {config['filters']}")
            if 'kernel_size' in config:
                print(f"  - Kernel size: {config['kernel_size']}")
            if 'units' in config:
                print(f"  - Units: {config['units']}")
            if 'rate' in config:
                print(f"  - Dropout rate: {config['rate']}")
    
    def get_weights(self):
        """Lấy weights của model"""
        if self.model is None:
            print("Vui lòng load model trước!")
            return None
        
        weights = self.model.get_weights()
        print(f"\n✓ Model có {len(weights)} weight tensors")
        
        for i, w in enumerate(weights):
            print(f"Weight {i+1}: shape {w.shape}, dtype {w.dtype}")
        
        return weights
    
    def predict_sample(self, input_data):
        """
        Dự đoán với input mẫu
        
        Args:
            input_data: numpy array với shape phù hợp với input của model
        """
        if self.model is None:
            print("Vui lòng load model trước!")
            return None
        
        try:
            prediction = self.model.predict(input_data)
            return prediction
        except Exception as e:
            print(f"✗ Lỗi khi dự đoán: {e}")
            return None
    
    def read_h5_structure(self):
        """Đọc cấu trúc file .h5 trực tiếp"""
        print("\n" + "="*60)
        print("CẤU TRÚC FILE .H5")
        print("="*60)
        
        def print_structure(name, obj):
            print(name)
            
        with h5py.File(self.model_path, 'r') as f:
            f.visititems(print_structure)
    
    def export_architecture(self, output_path='model_architecture.json'):
        """Xuất kiến trúc model ra file JSON"""
        if self.model is None:
            print("Vui lòng load model trước!")
            return
        
        try:
            model_json = self.model.to_json()
            with open(output_path, 'w') as f:
                json.dump(json.loads(model_json), f, indent=2)
            print(f"✓ Đã xuất kiến trúc model ra: {output_path}")
        except Exception as e:
            print(f"✗ Lỗi khi xuất kiến trúc: {e}")


# ============================
# CÁCH SỬ DỤNG
# ============================

if __name__ == "__main__":
    # Thay đổi đường dẫn file .h5 của bạn ở đây
    # Ví dụ các đường dẫn có thể:
    # MODEL_PATH = "model.h5"
    # MODEL_PATH = "models/drowsiness_model.h5"
    # MODEL_PATH = r"D:\PROJECTDriverDrowsiness\Final\model.h5"
    
    import os
    import glob
    
    # Tự động tìm file .h5 trong thư mục hiện tại
    h5_files = glob.glob("*.h5")
    
    if not h5_files:
        print("="*60)
        print("KHÔNG TÌM THẤY FILE .H5")
        print("="*60)
        print(f"Thư mục hiện tại: {os.getcwd()}")
        print("\nCác file trong thư mục:")
        for file in os.listdir():
            print(f"  - {file}")
        print("\n⚠️  Vui lòng:")
        print("1. Kiểm tra file .h5 có trong thư mục không")
        print("2. Hoặc sửa MODEL_PATH thành đường dẫn đầy đủ của file")
        print("\nVí dụ: MODEL_PATH = r'D:\\path\\to\\your\\model.h5'")
        exit()
    
    # Sử dụng file .h5 đầu tiên tìm được
    MODEL_PATH = "my_model.h5"
    
    print("="*60)
    print(f"Tìm thấy file: {MODEL_PATH}")
    print("="*60)
    
    # Khởi tạo reader
    reader = DrowsinessModelReader(MODEL_PATH)
    
    # 1. Load model
    model = reader.load_model()
    
    if model is not None:
        # 2. Xem tổng quan model
        reader.get_model_summary()
        
        # 3. Xem thông tin chi tiết
        reader.get_model_info()
        
        # 4. Xem thông tin từng layer
        reader.get_layer_info()
        
        # 5. Xem weights
        reader.get_weights()
        
        # 6. Đọc cấu trúc file .h5
        reader.read_h5_structure()
        
        # 7. Xuất kiến trúc ra JSON
        reader.export_architecture()
        
        # 8. Ví dụ dự đoán (cần điều chỉnh shape theo model của bạn)
        # input_shape = model.input_shape[1:]  # Bỏ batch dimension
        # sample_input = np.random.random((1,) + input_shape)
        # prediction = reader.predict_sample(sample_input)
        # print(f"\nKết quả dự đoán mẫu: {prediction}")