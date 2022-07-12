# OCR Extract Info from Vietnamese Identity Cards
This is an integrated system of **advanced image processing methods in image preprocessing** and the application of **Tesseract (Open Source OCR Engine)**.
Gives about *90-92% accuracy manually evaluated on 300 images*

---
**The current model has the following functions:**
- Adjust the size and drag the 4 sides perpendicularly even when taking photos at an angle (no more than 45 degrees)
- Self-recognize both sides of an ID card (new and old model)
- Crop the part of the photo containing the information (text) on the identity card
- Extract identifiable information into text

<center><img src="https://github.com/pdtlong/pdtlong.github.io/blob/main/images/ocr.gif" width="550"/></center>

## Libraries to run the model:
- Python 3.7
- pytesseract
- opencv2
- matplotlib
- numpy

---
### How to run?
Open jupyter file Test_4mat.ipynb
