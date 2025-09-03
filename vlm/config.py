import os

# VLM服务端口环境变量配置
VLM_GROUNDING_DINO_PORT = int(os.environ.get('VLM_GROUNDING_DINO_PORT', 12181))
VLM_BLIP2ITM_PORT = int(os.environ.get('VLM_BLIP2ITM_PORT', 12182))
VLM_MOBILE_SAM_PORT = int(os.environ.get('VLM_MOBILE_SAM_PORT', 12283))
VLM_YOLOV7_PORT = int(os.environ.get('VLM_YOLOV7_PORT', 12184))