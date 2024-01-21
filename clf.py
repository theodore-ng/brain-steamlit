# from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor

import json
import requests

# Run inference on an image
url = "https://api.ultralytics.com/v1/predict/PTHfVv9rwxuv8OsvV0aT"
headers = {"x-api-key": "51d786a5cd3a23433d3c99adb0b4f075aa9173291e"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open("./test/images/47_jpg.rf.ad1a9d52840cca4d53ff6403c1bf8512.jpg", "rb") as f:
	response = requests.post(url, headers=headers, data=data, files={"image": f})

# Check for successful response
# response.raise_for_status()

# Print inference results
# print(json.dumps(response.json(), indent=2))

# class BrainDetect:
#     def __init__(self, max_length=14, num_beams=4):
#         self.model = VisionEncoderDecoderModel.from_pretrained(
#             "nlpconnect/vit-gpt2-image-captioning"
#         )
#         self.feature_extractor = ViTFeatureExtractor.from_pretrained(
#             "nlpconnect/vit-gpt2-image-captioning"
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "nlpconnect/vit-gpt2-image-captioning"
#         )
#         self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

#     def predict(self, image):
#         images = []
#         images.append(image)
#         pixel_values = self.feature_extractor(
#             images=images, return_tensors="pt"
#         ).pixel_values
#         output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
#         preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#         preds = [pred.strip() for pred in preds]
#         return preds

