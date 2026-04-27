import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import os
import time
import tensorrt as trt # NUEVO: Importar TensorRT

from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory

from sarnet_py.util import get_palette

class SARNetSegmentation(Node):
    def __init__(self):
        super().__init__('zed_sarnet_segmentation')
        
        self.n_class = 12 
        self.input_size = (640, 480) 
        self.bridge = CvBridge()
        self.palette = get_palette() 
        self.device = torch.device("cuda")
        
        self.get_logger().info("Inicializando TensorRT...")

        # --- CARGA DEL MOTOR TENSORRT ---
        package_share_directory = get_package_share_directory('sarnet_py')
        engine_path = os.path.join(package_share_directory, 'weights', 'sarnet_fp16.engine')
        
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.trt_context = self.engine.create_execution_context()
        
        # Pre-alojar memoria en la GPU para la salida (ahorra tiempo en cada frame)
        # Sabiendo que la salida es (1, 12, 480, 640) en formato FP16
        self.output_tensor = torch.empty((1, self.n_class, self.input_size[1], self.input_size[0]), dtype=torch.float16, device=self.device).contiguous()
        self.get_logger().info("Motor TensorRT cargado y listo para volar.")

        # Control de FPS y sincronización
        self.is_processing = False
        self.target_fps = 30.0  # ¡Subimos el límite para probar la velocidad real!
        self.target_period = 1.0 / self.target_fps
        self.last_process_time = 0.0

        self.latest_depth_msg = None
        self.latest_info_msg = None

        qos_profile_REL = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.info_sub = self.create_subscription(CameraInfo, '/zed/zed_node/rgb/color/rect/camera_info', self.info_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(ROSImage, '/zed/zed_node/depth/depth_registered', self.depth_callback, qos_profile_REL)
        self.rgb_sub = self.create_subscription(ROSImage, '/zed/zed_node/rgb/color/rect/image', self.rgb_callback, qos_profile_sensor_data)
        
        self.pub = self.create_publisher(ROSImage, '/sarnet/mask', 1)

    def info_callback(self, msg):
        self.latest_info_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def rgb_callback(self, rgb_msg):
        if self.latest_depth_msg is None or self.latest_info_msg is None:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if (current_time - self.last_process_time) < self.target_period:
            return

        if self.is_processing:
            return
            
        self.is_processing = True
        self.last_process_time = current_time 
        start_total = time.time()
        
        # 1. Preprocesamiento (CPU a GPU)
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb).resize(self.input_size)
        
        # 1. Empaquetamos la memoria RAM para que sea un bloque continuo
        img_np = np.ascontiguousarray(np.array(pil_img).transpose(2,0,1))

        # 2. Pasamos a Tensor, subimos a GPU, convertimos a 16 bits y normalizamos
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device).half().div(255.0).contiguous()

        # 2. INFERENCIA TENSORRT (Magia)
        torch.cuda.synchronize()
        start_inference = time.time()
        
        # Vinculamos los punteros de memoria de PyTorch (Entrada y Salida) directamente a TensorRT
        bindings = [int(img_tensor.data_ptr()), int(self.output_tensor.data_ptr())]
        self.trt_context.execute_v2(bindings=bindings)
        
        torch.cuda.synchronize()
        inference_time = (time.time() - start_inference) * 1000
        
        # 3. Postprocesamiento (argmax)
        pred = self.output_tensor.argmax(1).squeeze(0).cpu().numpy()

        # Colorear máscara
        mask_color = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            mask_color[pred == i] = color

        # Procesamiento de Profundidad (Resto del código sin cambios)
        depth_image_raw = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding="passthrough")
        depth_image = cv2.resize(depth_image_raw, self.input_size, interpolation=cv2.INTER_NEAREST)
        
        fx, fy = self.latest_info_msg.k[0], self.latest_info_msg.k[4]
        cx, cy = self.latest_info_msg.k[2], self.latest_info_msg.k[5]

        first_responder_class_id = 1
        civilian_class_id = 2 

        people_mask = np.logical_or(pred == first_responder_class_id, pred == civilian_class_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(people_mask, connectivity=8)

        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < 225:
                continue

            person_mask = (labels == label_id)
            total_person_pixels = np.sum(person_mask)
            civilian_pixels = np.sum(np.logical_and(person_mask, pred == civilian_class_id))
            ratio_civil = civilian_pixels / total_person_pixels

            if ratio_civil > 0.20:
                texto_etiqueta = "VICTIMA"
                color_hud = (217, 67, 224) 
            else:
                texto_etiqueta = "RESCATISTA"
                color_hud = (0, 130, 255)  

            center_x = int(centroids[label_id][0])
            center_y = int(centroids[label_id][1])
            Z = np.nanmedian(depth_image[person_mask])

            if not np.isnan(Z) and not np.isinf(Z) and Z > 0:
                X = (center_x - cx) * Z / fx
                Y = (center_y - cy) * Z / fy
                distancia_real = np.sqrt(X**2 + Y**2 + Z**2)
                angulo_h = np.degrees(np.arctan2(X, Z))

                cv2.circle(mask_color, (center_x, center_y), 5, (255, 255, 255), -1)
                cv2.circle(mask_color, (center_x, center_y), 10, color_hud, 2)
                texto_final = f"{texto_etiqueta} D:{distancia_real:.2f}m A:{angulo_h:.1f}deg"
                cv2.putText(mask_color, texto_final, (center_x - 80, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        self.pub.publish(self.bridge.cv2_to_imgmsg(mask_color, "rgb8"))
        self.is_processing = False

        total_time = (time.time() - start_total) * 1000
        # self.get_logger().info(f"TIMER: Inferencia TRT: {inference_time:.2f}ms | Ciclo Total: {total_time:.2f}ms")

def main():
    rclpy.init()
    node = SARNetSegmentation()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()