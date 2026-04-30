import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

import torch
import numpy as np
import cv2
import os
import time
import tensorrt as trt

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ament_index_python.packages import get_package_share_directory

from sarnet_py.util import get_palette


class SARNetSegmentationDatasetBenchmark(Node):
    def __init__(self):
        super().__init__('zed_sarnet_segmentation_v5_dataset_benchmark')

        # ------------------------------------------------------------
        # PARÁMETROS
        # ------------------------------------------------------------
        self.declare_parameter('input_topic', '/sarnet/dataset/image')
        self.declare_parameter('output_topic', '/sarnet/mask')
        self.declare_parameter('publish_colored_mask', True)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.publish_colored_mask = bool(
            self.get_parameter('publish_colored_mask').value
        )

        # ------------------------------------------------------------
        # CONFIGURACIÓN GENERAL
        # ------------------------------------------------------------
        self.n_class = 12
        self.input_size = (640, 480)  # (ancho, alto)
        self.device = torch.device("cuda")

        self.bridge = CvBridge()

        # Paleta en formato NumPy para colorear la máscara de forma vectorizada
        self.palette = get_palette()
        self.palette_np = np.array(self.palette, dtype=np.uint8)

        # Contador interno de FPS
        self.frame_count = 0
        self.fps_t0 = time.time()
        self.log_every_n_frames = 100

        # Evita solapar inferencias
        self.is_processing = False

        self.get_logger().info("Inicializando nodo SARNet TensorRT v5 con dataset...")

        # ------------------------------------------------------------
        # CARGA DEL MOTOR TENSORRT
        # ------------------------------------------------------------
        package_share_directory = get_package_share_directory('sarnet_py')
        engine_path = os.path.join(
            package_share_directory,
            'weights',
            'sarnet_fp16.engine'
        )

        self.get_logger().info(f"Cargando engine TensorRT desde: {engine_path}")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("No se pudo deserializar el engine TensorRT.")

        self.trt_context = self.engine.create_execution_context()

        if self.trt_context is None:
            raise RuntimeError("No se pudo crear el contexto de ejecución TensorRT.")

        # ------------------------------------------------------------
        # PREASIGNACIÓN DE MEMORIA DE SALIDA EN GPU
        # ------------------------------------------------------------
        self.output_tensor = torch.empty(
            (1, self.n_class, self.input_size[1], self.input_size[0]),
            dtype=torch.float16,
            device=self.device
        ).contiguous()

        self.get_logger().info(
            f"Tensor de salida preasignado en GPU: {tuple(self.output_tensor.shape)}"
        )

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ------------------------------------------------------------
        # SUSCRIPCIÓN A IMÁGENES DEL DATASET
        # ------------------------------------------------------------
        self.rgb_sub = self.create_subscription(
            ROSImage,
            self.input_topic,
            self.rgb_callback,
            qos_sensor
        )

        self.pub = self.create_publisher(
            ROSImage,
            self.output_topic,
            qos_sensor
        )

        self.get_logger().info("Nodo v5 benchmark listo.")
        self.get_logger().info(f"Entrada: {self.input_topic}")
        self.get_logger().info(f"Salida:  {self.output_topic}")
        self.get_logger().info(
            "Modo: TensorRT FP16 con imágenes de dataset, "
            "sin ZED, sin profundidad, sin HUD y sin limitador de FPS."
        )

    def rgb_callback(self, rgb_msg):
        if self.is_processing:
            return

        self.is_processing = True

        try:
            # --------------------------------------------------------
            # 1. PREPROCESAMIENTO
            # --------------------------------------------------------
            # El publicador ya manda rgb8 y, normalmente, 640x480.
            img_rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="rgb8"
            )

            # Seguridad por si alguna imagen llega con otro tamaño
            if (
                img_rgb.shape[1] != self.input_size[0]
                or img_rgb.shape[0] != self.input_size[1]
            ):
                img_rgb = cv2.resize(
                    img_rgb,
                    self.input_size,
                    interpolation=cv2.INTER_LINEAR
                )

            # HWC -> CHW
            img_np = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))

            # NumPy CPU -> Tensor GPU FP16
            # Forma final: (1, 3, 480, 640)
            img_tensor = torch.from_numpy(img_np)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(
                device=self.device,
                dtype=torch.float16,
                non_blocking=True
            )
            img_tensor = img_tensor.div_(255.0).contiguous()

            # --------------------------------------------------------
            # 2. INFERENCIA TENSORRT
            # --------------------------------------------------------
            bindings = [
                int(img_tensor.data_ptr()),
                int(self.output_tensor.data_ptr())
            ]

            self.trt_context.execute_v2(bindings=bindings)

            # --------------------------------------------------------
            # 3. POSTPROCESAMIENTO MÍNIMO
            # --------------------------------------------------------
            pred = (
                self.output_tensor
                .argmax(dim=1)
                .squeeze(0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            # --------------------------------------------------------
            # 4. PUBLICACIÓN DE LA MÁSCARA
            # --------------------------------------------------------
            if self.publish_colored_mask:
                mask_color = self.palette_np[pred]
                mask_color = np.ascontiguousarray(mask_color)

                msg_out = self.bridge.cv2_to_imgmsg(
                    mask_color,
                    encoding="rgb8"
                )
            else:
                pred = np.ascontiguousarray(pred)

                msg_out = self.bridge.cv2_to_imgmsg(
                    pred,
                    encoding="mono8"
                )

            msg_out.header = rgb_msg.header

            self.pub.publish(msg_out)

            # --------------------------------------------------------
            # 5. FPS INTERNO
            # --------------------------------------------------------
            self.frame_count += 1

            if self.frame_count % self.log_every_n_frames == 0:
                now = time.time()
                elapsed = now - self.fps_t0
                fps = self.log_every_n_frames / elapsed

                self.get_logger().info(
                    f"FPS internos publicados en {self.output_topic}: {fps:.2f}"
                )

                self.fps_t0 = now

        except Exception as e:
            self.get_logger().error(f"Error en rgb_callback v5: {e}")

        finally:
            self.is_processing = False


def main(args=None):
    rclpy.init(args=args)

    node = SARNetSegmentationDatasetBenchmark()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()