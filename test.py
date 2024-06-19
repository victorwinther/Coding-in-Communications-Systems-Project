from PIL import Image
from PIL import ExifTags

img = Image.open('profile.jpg')
print(img.format)

img2 = Image.open('profile2.jpg')
print(img2.format)

print(img.getexif())
print(img2.getexif())

# Create a new EXIF metadata dictionary
exif_data = {
    ExifTags.TAG_RESOLUTION_UNIT: 2,  # Inches
    ExifTags.TAG_X_RESOLUTION: (72, 1),  # 72 dpi
    ExifTags.TAG_Y_RESOLUTION: (72, 1),  # 72 dpi
    ExifTags.TAG_Y_CB_CR_POSITIONING: 1,  # Centered
    ExifTags.TAG_SOFTWARE: 'Pillow',  # Software used (optional)
    # Add more EXIF tags as needed
}