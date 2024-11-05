import gdown

# Correct file ID after changing permission to 'Anyone with the link'
file_id = "1RA8PEMPKkIFHuM6XS0ptps6AHoc-6OgK"
output = "model.pth"  # Desired output filename

gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# https://drive.google.com/file/d/1RA8PEMPKkIFHuM6XS0ptps6AHoc-6OgK/view?usp=sharing