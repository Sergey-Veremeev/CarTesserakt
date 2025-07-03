import cv2
import pytesseract
import matplotlib.pyplot as plt

# Укажите путь к Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\veremeev\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def open_img(img_path):
    """
    Загружает изображение по пути и конвертирует в grayscale.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def carplate_extract(image, cascade_path):
    """
    Обнаруживает номерной знак на изображении и возвращает его обрезанную область.
    """
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError(f"Не удалось загрузить каскад: {cascade_path}")
    
    rects = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    
    if len(rects) == 0:
        print("Объекты не найдены.")
        return None
    
    # Выбираем первый найденный объект
    x, y, w, h = rects[1]
    
    # Обрезка области с учетом смещений
    cropped_img = image[y+5:y+h-5, x+2:x+w-8]
    
    return cropped_img

def preprocess_image(image):
    """
    Улучшает изображение для OCR: бинаризация и увеличение.
    """
    # Бинаризация с помощью Otsu
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Увеличение изображения для повышения точности OCR
    scale_percent = 150  # увеличение на 150%
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized

def main():
    img_path = '001.jpg'
    cascade_path = 'haarcascade_russian_plate_number.xml'
    
    try:
        gray_image = open_img(img_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    try:
        plate_img = carplate_extract(gray_image, cascade_path)
        if plate_img is None:
            print("Номерной знак не обнаружен.")
            return
    except Exception as e:
        print(e)
        return
    
    processed_img = preprocess_image(plate_img)
    
    # Визуализация подготовленного изображения
    plt.imshow(processed_img, cmap='gray')
    plt.axis('off')
    plt.show()
    
     # Распознавание текста с расширенным whitelist и режимом psm 8
    custom_config = (
         '--psm 8 '
         '--oem 3 '
         '-c tessedit_char_whitelist=АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЭЮЯabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
     )
     
    text = pytesseract.image_to_string(processed_img, config=custom_config)
     
    print('Распознанный текст:', text.strip())

if __name__ == '__main__':
     main()
