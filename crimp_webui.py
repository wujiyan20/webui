import PIL.Image as Image
import gradio as gr

from ultralytics import YOLO
import torch 
print(torch.cuda.is_available())

import cv2
from PIL import ImageDraw, ImageFont
import numpy

model = YOLO("best_wire_segment.pt")


def mask_image(image, x_coordinates, y_coordinates, transparency):

    im = image
    draw = ImageDraw.Draw(im)

    pixel_coordinates = [(int(x * im.width), int(y * im.height)) for x, y in zip(x_coordinates, y_coordinates)]

    mask = Image.new('RGBA', im.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)

    # Draw the polygon
    mask_draw.polygon(pixel_coordinates, fill=(255, 0, 0, int(255 * transparency)), width=1)

    # Composite the original image and the masked image
    masked_image = Image.alpha_composite(im.convert('RGBA'), mask)
    
    return masked_image
    
def draw_text(image, polygon_area, image_width, image_height):

    draw = ImageDraw.Draw(image)

    center_position = (image_width // 2, image_height // 2)

    font_size = 60
    font = ImageFont.truetype("arial.ttf", font_size)  # You can replace "arial.ttf" with the path to your preferred font file

    area_text = f"Area: " + str(round(polygon_area, 3)) + " mm2"
    _, _, text_width, text_height = draw.textbbox((0,0), area_text, font=font)
    text_position = (center_position[0] - text_width // 2, center_position[1] - (image_height)//2.5)

    draw.text(text_position, area_text, fill='red', font=font)

def predict_image(img, conf_threshold, iou_threshold):
    
    multiplier = (5/815)**2 # fixed value here, need to change    
    
    image_width, image_height = img.size
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
        device = "0"
    )
    segment_mask = results[0].masks[0].xyn[0]
    x = []
    y = []
    for point in segment_mask:
        x.append(point[0])
        y.append(point[1])
    segment_mask[:, 0] = segment_mask[:, 0] * image_height
    segment_mask[:, 1] = segment_mask[:, 1] * image_width
    
    polygon_area = cv2.contourArea(segment_mask) * multiplier
    masked_image = mask_image(img, x, y, 0.3)
    draw_text(masked_image, polygon_area, image_width, image_height)
    im = numpy.array(masked_image)
    # im = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Crimping AI Software v1.0",
    description="Upload crimping cross section images (Wire/Insulation) for inference.",
    examples=[
        ['./assets/0b469a14-wb2_15x.jpg', 0.25, 0.45],
        ["./assets/1a4556e2-wb_20x_9.jpg", 0.25, 0.45],
        ["./assets/7a098f35-wb_20x_9.jpg", 0.25, 0.45],
        ["./assets/8b782a44-wb2_15x_2.jpg", 0.25, 0.45],
    ]
)

if __name__ == '__main__':
    iface.launch(server_name = "0.0.0.0", server_port = 9000, share=False)