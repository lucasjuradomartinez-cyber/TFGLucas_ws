import torch
import os
from sarnet_py.U_Net_SE_V2 import UNet

def main():
    # 1. Configuración
    device = torch.device("cuda")
    n_class = 12
    weights_path = 'weights/checkpoint_desde0_v3_0.7033_0.3449.pt'
    onnx_path = 'weights/sarnet_fp16.onnx'

    print("Cargando modelo PyTorch...")
    model = UNet(n_class).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 2. Convertir el modelo a FP16 (Mitad de precisión) para máxima velocidad
    model.half()

    # 3. Crear un tensor "falso" (dummy) del tamaño exacto de tus imágenes
    # Formato: (Batch_size, Canales, Alto, Ancho)
    dummy_input = torch.randn(1, 3, 480, 640, device=device).half()

    print("Exportando a ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=11,          # Versión estable para TensorRT
        do_constant_folding=True,  # Optimiza constantes matemáticas
        input_names=['input'],     # Nombre de la capa de entrada
        output_names=['output']    # Nombre de la capa de salida
    )
    print(f"¡Éxito! Modelo ONNX guardado en: {onnx_path}")

if __name__ == '__main__':
    main()
