import os
import glob
import random
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class DatasetImagePublisherV5(Node):
    def __init__(self):
        super().__init__('dataset_image_publisher_v5')

        
        # PARÁMETROS
        self.declare_parameter(
            'dataset_dir',
            os.path.expanduser('~/TFGLucas_ws/SARNet_dataset/img/train')
        )
        self.declare_parameter('output_topic', '/sarnet/dataset/image')
        self.declare_parameter('publish_fps', 120.0)
        self.declare_parameter('random_order', False)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        self.dataset_dir = self.get_parameter('dataset_dir').value
        self.output_topic = self.get_parameter('output_topic').value
        self.publish_fps = float(self.get_parameter('publish_fps').value)
        self.random_order = bool(self.get_parameter('random_order').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)

        self.bridge = CvBridge()

        # QoS
        # BEST_EFFORT + depth=1:
        # Si el consumidor no llega a leer todo, no se acumulan imágenes antiguas.
        # Siempre interesa que esté disponible la imagen más reciente.
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub = self.create_publisher(
            ROSImage,
            self.output_topic,
            qos_sensor
        )

        
        # CARGA DEL DATASET EN RAM
        self.image_msgs = []
        self.index = 0

        self.get_logger().info(f"Cargando imágenes desde: {self.dataset_dir}")
        self._load_dataset_images()

        if len(self.image_msgs) == 0:
            raise RuntimeError(
                f"No se encontraron imágenes válidas en: {self.dataset_dir}"
            )

        self.get_logger().info(
            f"Dataset cargado en RAM: {len(self.image_msgs)} imágenes "
            f"redimensionadas a {self.width}x{self.height}"
        )

        self.get_logger().info(f"Publicando en: {self.output_topic}")

        if self.publish_fps <= 0.0:
            self.get_logger().warn(
                "publish_fps <= 0: publicando sin límite. "
                "Esto puede consumir mucha CPU."
            )
        else:
            self.get_logger().info(
                f"Publicando a {self.publish_fps:.1f} FPS teóricos."
            )

        # Contador de FPS del publicador
        self.pub_count = 0
        self.pub_t0 = time.time()
        self.log_every_n_frames = 500

    def _load_dataset_images(self):
        extensions = [
            '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF'
        ]

        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(self.dataset_dir, ext)))

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            self.get_logger().error(
                f"No se han encontrado imágenes en {self.dataset_dir}"
            )
            return

        for path in image_paths:
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)

            if img_bgr is None:
                self.get_logger().warn(f"No se pudo leer la imagen: {path}")
                continue

            # Redimensionamos aquí para que el nodo de inferencia no tenga que hacerlo
            img_bgr = cv2.resize(
                img_bgr,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR
            )

            # La red trabaja con RGB. Publicamos directamente en rgb8.
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding='rgb8')
            msg.header.frame_id = 'dataset'

            self.image_msgs.append(msg)

    def publish_next(self):
        if self.random_order:
            msg = random.choice(self.image_msgs)
        else:
            msg = self.image_msgs[self.index]
            self.index += 1

            if self.index >= len(self.image_msgs):
                self.index = 0

        # Actualizamos timestamp en cada publicación
        msg.header.stamp = self.get_clock().now().to_msg()

        self.pub.publish(msg)

        self.pub_count += 1

        if self.pub_count % self.log_every_n_frames == 0:
            now = time.time()
            elapsed = now - self.pub_t0
            fps = self.log_every_n_frames / elapsed

            self.get_logger().info(
                f"FPS publicados en {self.output_topic}: {fps:.2f}"
            )

            self.pub_t0 = now


def main(args=None):
    rclpy.init(args=args)

    node = DatasetImagePublisherV5()

    try:
        if node.publish_fps <= 0.0:
            # Publicación sin límite
            while rclpy.ok():
                node.publish_next()
                rclpy.spin_once(node, timeout_sec=0.0)
        else:
            # Publicación con frecuencia fija
            period = 1.0 / node.publish_fps

            while rclpy.ok():
                t0 = time.time()
                node.publish_next()
                rclpy.spin_once(node, timeout_sec=0.0)
                elapsed = time.time() - t0
                sleep_time = period - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()